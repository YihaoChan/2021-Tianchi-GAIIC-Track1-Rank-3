# coding=utf-8
"""Run masked LM/next sentence masked_lm pre-training for BERT."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from seed import seed
import os
import time
import modeling
import optimization
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')

flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

# Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("horovod", False, "Whether to use Horovod for multi-gpu runs")

flags.DEFINE_bool("report_loss", False, "Whether to report total loss during training.")

flags.DEFINE_bool("manual_fp16", False, "Whether to use fp32 or fp16 arithmetic on GPU. "
                                        "Manual casting is done instead of using AMP")

flags.DEFINE_bool("use_xla", False, "Whether to enable XLA JIT compilation.")

flags.DEFINE_bool("use_fp16", False, "Whether to enable AMP ops.")

flags.DEFINE_string("optimizer", 'lamb', "The optimizer.")


# report samples/sec, total loss and learning rate during training
class _LogSessionRunHook(tf.train.SessionRunHook):
    def __init__(self, global_batch_size, display_every=10, hvd_rank=-1):
        self.global_batch_size = global_batch_size
        self.display_every = display_every
        self.hvd_rank = hvd_rank

    def after_create_session(self, session, coord):
        self.elapsed_secs = 0.
        self.count = 0

    def before_run(self, run_context):
        self.t0 = time.time()
        if FLAGS.manual_fp16 or FLAGS.use_fp16:
            return tf.train.SessionRunArgs(
                fetches=['step_update:0', 'total_loss:0',
                         'learning_rate:0', 'nsp_loss:0',
                         'mlm_loss:0', 'loss_scale:0'])
        else:
            return tf.train.SessionRunArgs(
                fetches=['step_update:0', 'total_loss:0',
                         'learning_rate:0', 'nsp_loss:0',
                         'mlm_loss:0'])

    def after_run(self, run_context, run_values):
        self.elapsed_secs += time.time() - self.t0
        self.count += 1
        if FLAGS.manual_fp16 or FLAGS.use_fp16:
            global_step, total_loss, lr, nsp_loss, mlm_loss, loss_scaler = run_values.results
        else:
            global_step, total_loss, lr, nsp_loss, mlm_loss = run_values.results
        print_step = global_step + 1  # One-based index for printing.
        if print_step == 1 or print_step % self.display_every == 0:
            dt = self.elapsed_secs / self.count
            img_per_sec = self.global_batch_size / dt
            if self.hvd_rank >= 0:
                if FLAGS.manual_fp16 or FLAGS.use_fp16:
                    print(
                        'Rank = %2d :: Step = %6i Throughput = %11.1f MLM Loss = %10.4e NSP Loss = %10.4e Loss = %6.3f LR = %6.4e Loss scale = %6.4e' %
                        (self.hvd_rank, print_step, img_per_sec, mlm_loss, nsp_loss, total_loss, lr, loss_scaler))
                else:
                    print(
                        'Rank = %2d :: Step = %6i Throughput = %11.1f MLM Loss = %10.4e NSP Loss = %10.4e Loss = %6.3f LR = %6.4e' %
                        (self.hvd_rank, print_step, img_per_sec, mlm_loss, nsp_loss, total_loss, lr))
            else:
                if FLAGS.manual_fp16 or FLAGS.use_fp16:
                    print(
                        'Step = %6i Throughput = %11.1f MLM Loss = %10.4e NSP Loss = %10.4e Loss = %6.3f LR = %6.4e Loss scale = %6.4e' %
                        (print_step, img_per_sec, mlm_loss, nsp_loss, total_loss, lr, loss_scaler))
                else:
                    print('Step = %6i Throughput = %11.1f MLM Loss = %10.4e NSP Loss = %10.4e Loss = %6.3f LR = %6.4e' %
                          (print_step, img_per_sec, mlm_loss, nsp_loss, total_loss, lr))
            self.elapsed_secs = 0.
            self.count = 0


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings, hvd=None):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]
        next_sentence_labels = features["next_sentence_labels"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
            compute_type=tf.float16 if FLAGS.manual_fp16 else tf.float32)

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
            bert_config, model.get_sequence_output(), model.get_embedding_table(),
            masked_lm_positions, masked_lm_ids,
            masked_lm_weights)

        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs) = get_next_sentence_output(
            bert_config, model.get_pooled_output(), next_sentence_labels)

        masked_lm_loss = tf.identity(masked_lm_loss, name="mlm_loss")
        next_sentence_loss = tf.identity(next_sentence_loss, name="nsp_loss")
        total_loss = masked_lm_loss + next_sentence_loss
        total_loss = tf.identity(total_loss, name='total_loss')

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        if init_checkpoint and (hvd is None or hvd.rank() == 0):
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  %d :: name = %s, shape = %s%s", 0 if hvd is None else hvd.rank(), var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                loss=total_loss, init_lr=learning_rate, num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                hvd=hvd, manual_fp16=FLAGS.manual_fp16, use_fp16=FLAGS.use_fp16, optimizer_algo=FLAGS.optimizer)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                          masked_lm_weights, next_sentence_example_loss,
                          next_sentence_log_probs, next_sentence_labels):
                """Computes the loss and accuracy of the model."""
                masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                                 [-1, masked_lm_log_probs.shape[-1]])
                masked_lm_predictions = tf.argmax(
                    masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = tf.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights)
                masked_lm_mean_loss = tf.metrics.mean(
                    values=masked_lm_example_loss, weights=masked_lm_weights)

                next_sentence_log_probs = tf.reshape(
                    next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
                next_sentence_predictions = tf.argmax(
                    next_sentence_log_probs, axis=-1, output_type=tf.int32)
                next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
                next_sentence_accuracy = tf.metrics.accuracy(
                    labels=next_sentence_labels, predictions=next_sentence_predictions)
                next_sentence_mean_loss = tf.metrics.mean(
                    values=next_sentence_example_loss)

                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss,
                    "next_sentence_accuracy": next_sentence_accuracy,
                    "next_sentence_loss": next_sentence_mean_loss,
                }

            eval_metric_ops = metric_fn(
                masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                masked_lm_weights, next_sentence_example_loss,
                next_sentence_log_probs, next_sentence_labels
            )
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metric_ops)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(tf.cast(input_tensor, tf.float32), output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
    """Get loss and log probs for the next sentence prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.variable_scope("cls/seq_relationship"):
        output_weights = tf.get_variable(
            "output_weights",
            shape=[2, bert_config.hidden_size],
            initializer=modeling.create_initializer(bert_config.initializer_range))
        output_bias = tf.get_variable(
            "output_bias", shape=[2], initializer=tf.zeros_initializer())

        logits = tf.matmul(tf.cast(input_tensor, tf.float32), output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files,
                     batch_size,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4,
                     hvd=None):
    """Creates an `input_fn` closure to be passed to Estimator."""

    def input_fn():
        """The actual input function."""

        name_to_features = {
            "input_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "next_sentence_labels":
                tf.FixedLenFeature([1], tf.int64),
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            if hvd is not None: d = d.shard(hvd.size(), hvd.rank())
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if FLAGS.use_fp16:
        os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"

    if FLAGS.horovod:
        import horovod.tensorflow as hvd
        hvd.init()

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Input Files ***")
    for input_file in input_files:
        tf.logging.info("  %s" % input_file)

    config = tf.ConfigProto()
    if FLAGS.horovod:
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        if len(input_files) < hvd.size():
            raise ValueError("Input Files must be sharded")
    if FLAGS.use_xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    config = tf.ConfigProto()
    if FLAGS.horovod:
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        config.gpu_options.allow_growth = True
    #    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    if FLAGS.use_xla: config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        session_config=config,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps if not FLAGS.horovod or hvd.rank() == 0 else None,
        # This variable controls how often estimator reports examples/sec.
        # Default value is every 100 steps.
        # When --report_loss is True, we set to very large value to prevent
        # default info reporting from estimator.
        # Ideally we should set it to None, but that does not work.
        log_step_count_steps=10000 if FLAGS.report_loss else 100)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate if not FLAGS.horovod else FLAGS.learning_rate * hvd.size(),
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_one_hot_embeddings=False,
        hvd=None if not FLAGS.horovod else hvd)

    training_hooks = []
    if FLAGS.horovod and hvd.size() > 1:
        training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))
    if FLAGS.report_loss:
        global_batch_size = FLAGS.train_batch_size if not FLAGS.horovod else FLAGS.train_batch_size * hvd.size()
        training_hooks.append(_LogSessionRunHook(global_batch_size, 1, -1 if not FLAGS.horovod else hvd.rank()))

    training_hooks = []
    if FLAGS.report_loss and (not FLAGS.horovod or hvd.rank() == 0):
        global_batch_size = FLAGS.train_batch_size if not FLAGS.horovod else FLAGS.train_batch_size * hvd.size()
        training_hooks.append(_LogSessionRunHook(global_batch_size, 100))
    if FLAGS.horovod:
        training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        train_input_fn = input_fn_builder(
            input_files=input_files,
            batch_size=FLAGS.train_batch_size,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=True,
            hvd=None if not FLAGS.horovod else hvd)
        estimator.train(input_fn=train_input_fn, hooks=training_hooks, max_steps=FLAGS.num_train_steps)

    if FLAGS.do_eval and (not FLAGS.horovod or hvd.rank() == 0):
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_input_fn = input_fn_builder(
            input_files=input_files,
            batch_size=FLAGS.eval_batch_size,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False,
            hvd=None if not FLAGS.horovod else hvd)

        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    seed()

    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    if FLAGS.use_xla and FLAGS.manual_fp16:
        print('WARNING! Combining --use_xla with --manual_fp16 may prevent convergence.')
        print('         This warning message will be removed when the underlying')
        print('         issues have been fixed and you are running a TF version')
        print('         that has that fix.')
    tf.app.run()
