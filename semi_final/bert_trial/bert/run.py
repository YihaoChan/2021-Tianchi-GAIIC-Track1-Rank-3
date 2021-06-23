import os
import directory


class RunConfig(object):
    def __init__(self):
        # 预训练步数
        self.num_train_steps = 30000

        # 预训练、训练的batch_size
        self.batch_size = 16

        # 联合训练的epoch
        self.joint_num_epochs = 10

        # 分开训练：任务1(区域)的epoch
        self.separate_region_num_epochs = 5

        # 分开训练：任务2(类型)的epoch
        self.separate_category_num_epochs = 5

        self.seq_len = 100

        self.n_splits = 5

        # 分开训练权重
        self.separate_weight = 0.7

        # 联合训练权重
        self.joint_weight = 0.3


def main():
    run_config = RunConfig()

    # 生成语料库和词典
    os.system('python3 ./bert/corpus_vocab.py')

    # 生成预训练数据
    os.system('python3 ./bert/create_pretraining_data.py \
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
    os.system('python3 ./bert/pretraining.py \
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
        directory.PRETRAINING_OUTPUT_PATH, directory.CHECKPOINT_PATH,
        directory.BERT_CONFIG_PATH, run_config.batch_size, run_config.seq_len, run_config.num_train_steps
    ))

    # 调用预训练权重后接MLP微调进行训练、预测
    os.system('python3 ./bert/separate_region_train_val.py')
    os.system('python3 ./bert/separate_region_predict.py')
    os.system('python3 ./bert/separate_category_train_val.py')
    os.system('python3 ./bert/separate_category_predict.py')
    os.system('python3 ./bert/separate_predict.py')
    os.system('python3 ./bert/joint_train_val.py')
    os.system('python3 ./bert/joint_predict.py')
    os.system('python3 ./bert/merge.py')


if __name__ == '__main__':
    main()
