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

# 复赛B榜测试集路径
SEMI_TEST_SET_B_PATH = DATASET_DIR + '/testB.csv'

# 中间文件目录
DATA_DIR = './nezha/data'

# Bert配置文件路径
BERT_CONFIG_PATH = './nezha/pretrain/bert_config.json'

# 语料库文本文件路径
CORPUS_PATH = DATA_DIR + '/corpus.txt'

# 词典文件路径
VOCAB_PATH = DATA_DIR + '/vocab.txt'

# 预训练数据输出文件路径
PRETRAINING_OUTPUT_PATH = DATA_DIR + '/tf_examples.tfrecord'

# 预训练模型保存路径
CHECKPOINT_PATH = DATA_DIR + '/checkpoint'

# 训练模型参数保存路径
MODEL_DIR = './nezha/model'

# 区域任务的结果文件
REGION_RESULT_PATH = DATA_DIR + '/separate_region_output.txt'

# 类型任务的结果文件
CATEGORY_RESULT_PATH = DATA_DIR + '/separate_category_output.txt'

# 分开训练：提交的csv文件路径
SEPARATE_RESULT_PATH = DATA_DIR + '/result_separate.csv'

# 联合训练：提交的csv文件路径
JOINT_RESULT_PATH = DATA_DIR + '/result_joint.csv'

# 最终提交的csv文件路径
SUBMISSION_PATH = './result_nezha.csv'
