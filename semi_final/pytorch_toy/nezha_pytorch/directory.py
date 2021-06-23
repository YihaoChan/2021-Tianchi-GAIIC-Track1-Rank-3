# 数据集存放目录
DATASET_DIR = '/tcdata'

# 初赛训练集路径
PREM_TRAIN_SET_PATH = DATASET_DIR + '/track1_round1_train_20210222.csv'

# 初赛A榜测试集路径
PREM_TEST_SET_A_PATH = DATASET_DIR + '/track1_round1_testA_20210222.csv'

# 初赛B榜测试集路径
PREM_TEST_SET_B_PATH = DATASET_DIR + '/track1_round1_testB.csv'

# 复赛训练集路径
SEMI_TRAIN_SET_PATH = DATASET_DIR + '/train.csv'

# 复赛A榜测试集路径
SEMI_TEST_SET_A_PATH = DATASET_DIR + '/testA.csv'

# 中间文件目录
DATA_DIR = './nezha_pytorch/data'

# 语料库文本文件路径
CORPUS_PATH = DATA_DIR + '/corpus.txt'

# Bert配置文件路径
BERT_CONFIG_PATH = './nezha_pytorch/helper/bert_config.json'

# 预训练文件目录
PRETRAIN_DIR = './nezha_pytorch/pretrained_model'

# 词典文件路径
VOCAB_PATH = PRETRAIN_DIR + '/vocab.txt'

# 训练模型目录
MODEL_DIR = './nezha_pytorch/model'

# 区域任务的结果文件
REGION_RESULT_PATH = DATA_DIR + '/separate_region_output.txt'

# 类型任务的结果文件
CATEGORY_RESULT_PATH = DATA_DIR + '/separate_category_output.txt'

# 提交的csv文件目录
SUBMISSION_PATH = './result.csv'
