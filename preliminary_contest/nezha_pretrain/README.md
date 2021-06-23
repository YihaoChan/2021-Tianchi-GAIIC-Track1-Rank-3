# NEZHA预训练模型

## 1 文件结构

```
├── bert_config.json
├── corpus_vocab.py----------------------------制作语料库和词典
├── create_pretraining_data.py
├── datasets------------------初赛数据集
│   ├── track1_round1_testA_20210222.csv
│   ├── track1_round1_testB.csv
│   └── track1_round1_train_20210222.csv
├── directory.py-------------------------------数据集、中间文件、生成文件等路径
├── gpu_environment.py
├── modeling.py
├── optimization.py
├── predict.py---------------------------------推理
├── preprocess.py------------------------------数据预处理
├── pretraining.py-----------------------------预训练
├── run.py-------------------------------------运行脚本
├── seed.py------------------------------------设定随机种子
├── tokenization.py
└── train_val.py-------------------------------训练
```

## 2 执行环境

操作系统：Ubuntu 20.04.1

cuda版本：11.0

Python版本：3.6

Python依赖：执行`pip3 install -r requirements.txt`命令安装。

## 3 实验结果

| 模型 | seq长度 | 预训练步数 | 训练fold | 训练epoch | 线上得分 |
| ------- | ---- | ---------- | ---------- | ---------- | ---------- |
| NEZHA | 100     | 30000 | 5   | 5  | 0.910 |
| NEZHA | 100     | 30000 | 5   | 10 | 0.901   |
