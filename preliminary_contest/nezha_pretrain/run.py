import os
import directory


class RunConfig(object):
    def __init__(self):
        self.num_epochs = 5

        self.batch_size = 16

        self.seq_len = 100

        self.n_splits = 5

        self.num_train_steps = 30000


def run():
    run_config = RunConfig()

    # 生成语料库和词典
    os.system('python3 ./corpus_vocab.py')

    # 生成训练数据
    os.system('python3 ./create_pretraining_data.py \
    --input_file=%s \
    --output_file=%s \
    --vocab_file=%s \
    --do_lower_case=True \
    --max_seq_length=%d \
    --max_predictions_per_seq=20 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=5' % (
        directory.CORPUS_PATH, directory.PRETRAINING_OUTPUT_PATH, directory.VOCAB_PATH, run_config.seq_len
    ))

    # 预训练
    os.system('python3 ./pretraining.py \
    --input_file=%s \
    --output_dir=%s \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=%s \
    --train_batch_size=%d \
    --max_seq_length=%d \
    --max_predictions_per_seq=20 \
    --num_train_steps=%d \
    --learning_rate=5e-5' % (
        directory.PRETRAINING_OUTPUT_PATH, directory.CHECKPOINT_PATH, directory.BERT_CONFIG_PATH,
        run_config.batch_size, run_config.seq_len, run_config.num_train_steps
    ))

    # 调用预训练权重后接MLP微调进行训练
    os.system('python3 ./train_val.py')

    # 预测
    os.system('python3 ./predict.py')


if __name__ == '__main__':
    run()
