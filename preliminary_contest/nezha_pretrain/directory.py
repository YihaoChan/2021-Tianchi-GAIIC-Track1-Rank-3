# 数据集存放目录
DATASET_DIR = './datasets'

# 训练集路径
TRAIN_SET_PATH = DATASET_DIR + '/track1_round1_train_20210222.csv'

# A榜测试集路径
TEST_SET_A_PATH = DATASET_DIR + '/track1_round1_testA_20210222.csv'

# B榜测试集路径
TEST_SET_B_PATH = DATASET_DIR + '/track1_round1_testB.csv'

# 中间文件目录
DATA_DIR = './data'

# Bert配置文件路径
BERT_CONFIG_PATH = './bert_config.json'

# 语料库文本文件路径
CORPUS_PATH = DATA_DIR + '/corpus.txt'

# 词典文件路径
VOCAB_PATH = DATA_DIR + '/vocab.txt'

# 预训练数据输出文件路径
PRETRAINING_OUTPUT_PATH = DATA_DIR + '/tf_examples.tfrecord'

# 预训练模型保存路径
CHECKPOINT_PATH = DATA_DIR + '/checkpoint'

# 训练模型参数保存路径
MODEL_DIR = './model'

# 提交的csv文件目录
SUBMISSION_DIR = './submission'

SUBMISSION_PATH = SUBMISSION_DIR + '/submission.csv'
