# Noah's Ark: ***N***EZHA Trained J***O***intly and Sep***A***rately with ***H***AN, L***S***TM and Deep Pyr***A***mid Convolutional Neu***R***al Networ***K***s

## 1 文件结构

```
├── dl-----------------基础模型部分
│   ├── code----------------------训练及推理
│   │   ├── test------------------------------推理
│   │   │   ├── cvs_fusion.py------------------------------基础模型结果融合
│   │   │   ├── label1
│   │   │   │   ├── config.py------------------------------模型及训练参数
│   │   │   │   ├── directory.py---------------------------数据集路径
│   │   │   │   ├── dpcnn_infer.py-------------------------DPCNN任务1推理
│   │   │   │   ├── dpcnn.py-------------------------------DPCNN模型定义
│   │   │   │   ├── han_infer.py---------------------------HAN任务1推理
│   │   │   │   ├── han.py---------------------------------HAN模型定义
│   │   │   │   ├── lstm_infer.py--------------------------LSTM任务1推理
│   │   │   │   ├── lstm.py--------------------------------LSTM模型定义
│   │   │   │   └── stopwords.txt--------------------------停用词
│   │   │   └── label2
│   │   │   │   ├── config.py------------------------------模型及训练参数
│   │   │   │   ├── directory.py---------------------------数据集路径
│   │   │   │   ├── dpcnn_infer.py-------------------------DPCNN任务2推理
│   │   │   │   ├── dpcnn.py-------------------------------DPCNN模型定义
│   │   │   │   ├── han_infer.py---------------------------HAN任务2推理
│   │   │   │   ├── han.py---------------------------------HAN模型定义
│   │   │   │   ├── lstm_infer.py--------------------------LSTM任务2推理
│   │   │   │   ├── lstm.py--------------------------------LSTM模型定义
│   │   │   │   └── stopwords.txt--------------------------停用词
│   │   └── train-----------------------------训练
│   │       ├── label1
│   │       │   ├── concat.py------------------------------拼接初赛和复赛训练集，提取任务1标签
│   │       │   ├── config.py------------------------------模型及训练参数
│   │       │   ├── directory.py---------------------------数据集路径
│   │       │   ├── dpcnn_datasets.py----------------------数据预处理
│   │       │   ├── dpcnn.py-------------------------------DPCNN模型定义
│   │       │   ├── dpcnn_train.py-------------------------DPCNN任务1训练
│   │       │   ├── han_datasets.py------------------------数据预处理
│   │       │   ├── han.py---------------------------------HAN模型定义
│   │       │   ├── han_train.py---------------------------HAN任务1训练
│   │       │   ├── lstm_datasets.py-----------------------数据预处理
│   │       │   ├── lstm.py--------------------------------LSTM模型定义
│   │       │   ├── lstm_train.py--------------------------LSTM任务1训练
│   │       │   ├── seed.py--------------------------------设定随机种子
│   │       │   └── stopwords.txt--------------------------停用词
│   │       └── label2
│   │       │   ├── config.py------------------------------模型及训练参数
│   │       │   ├── directory.py---------------------------数据集路径
│   │       │   ├── dpcnn_datasets.py----------------------数据预处理
│   │       │   ├── dpcnn.py-------------------------------DPCNN模型定义
│   │       │   ├── dpcnn_train.py-------------------------DPCNN任务2训练
│   │       │   ├── han_datasets.py------------------------数据预处理
│   │       │   ├── han.py---------------------------------HAN模型定义
│   │       │   ├── han_train.py---------------------------HAN任务2训练
│   │       │   ├── lstm_datasets.py-----------------------数据预处理
│   │       │   ├── lstm.py--------------------------------LSTM模型定义
│   │       │   ├── lstm_train.py--------------------------LSTM任务2训练
│   │       │   ├── seed.py--------------------------------设定随机种子
│   │       │   └── stopwords.txt--------------------------停用词
│   ├── prediction_result---------基础模型不同任务结果拼接
│   │   ├── dpcnnMerge2label.py----------------------------拼接DPCNN任务1和任务2的预测结果
│   │   ├── hanMerge2label.py------------------------------拼接HAN任务1和任务2的预测结果
│   │   ├── label1
│   │   ├── label2
│   │   └── lstmMerge2label.py-----------------------------拼接LSTM任务1和任务2的预测结果
│   └── user_data
│       └── model_data
│           ├── label1
│           └── label2
├── Dockerfile---------定制镜像
├── merge.py-----------基础模型及预训练模型结果融合
├── nezha--------------预训练模型部分
│   ├── corpus_vocab.py------------------------------------制作语料库和词典
│   ├── create_pretraining_data.py-------------------------构建预训练格式数据
│   ├── directory.py---------------------------------------数据集、中间文件、生成文件等路径
│   ├── helper
│   │   ├── adv_training.py--------------------------------Trick:对抗训练
│   │   ├── data_generator.py------------------------------数据生成器
│   │   ├── preprocess.py----------------------------------数据预处理
│   │   ├── seed.py----------------------------------------设定随机种子
│   │   └── warmup_cosine_decay.py-------------------------Trick:Warm Up + 余弦退火
│   ├── joint_predict.py-----------------------------------联合训练策略:推理
│   ├── joint_train_val.py---------------------------------联合训练策略:微调训练
│   ├── merge.py-------------------------------------------分开训练与联合训练结果融合
│   ├── nsp_corpus_vocab.py--------------------------------制作NSP任务的语料库和词典
│   ├── pretrain------------------预训练参数及辅助代码
│   │   ├── bert_config.json
│   │   ├── gpu_environment.py
│   │   ├── modeling.py
│   │   ├── optimization.py
│   │   └── tokenization.py
│   ├── pretraining.py-------------------------------------预训练
│   ├── run.py---------------------------------------------运行脚本
│   ├── separate_category_predict.py-----------------------分开训练策略:任务2推理
│   ├── separate_category_train_val.py---------------------分开训练策略:任务2微调训练
│   ├── separate_predict.py--------------------------------分开训练策略:拼接任务1和任务2的预测结果
│   ├── separate_region_predict.py-------------------------分开训练策略:任务1推理
│   └── separate_region_train_val.py-----------------------分开训练策略:任务1微调训练
└── run.sh--------------完整方案全流程运行脚本
```

## 2 线上环境

- 镜像地址：registry.cn-shanghai.aliyuncs.com/tcc-public/tensorflow:1.13.1-cuda10.0-py3

- 操作系统：Ubuntu 18.04

- 显卡及显存：NVIDIA V100 (16GB)

- CUDA版本：10.0

- Python版本：3.5.2

- Python依赖：

  ```
  bert4keras==0.10.0
  keras==2.3.1
  tensorflow-gpu==1.15.0
  scikit-learn==0.22.2
  torch==1.4.0
  h5py==2.10.0
  pandas==0.24.2
  iterative-stratification==0.1.6
  tqdm==4.60.0
  ```

## 3 实验结果

### 3.1 A榜

#### 3.1.1 NEZHA & BERT

| 模型 | 训练策略 | 余弦退火 | seq长度 | 预训练步数 | 训练fold | 训练epoch | 任务1得分 | 任务2得分 | 总得分 |
| ---- | ---------- | ---------- | ---------- | ---------- | ------- | ------- | ------- | ------- | ---- |
| NEZHA | 分开:0.7<br/>联合:0.3 | 是 | 100 | 30000 | 8 | 联合:10<br/>分开:5 | 0.9398 | 0.9388 | 0.9394 |
| NEZHA:0.7<br>BERT:0.3 | 分开:0.7<br/>联合:0.3 | 是 | 100 | 30000 | 5 | 联合:10<br>分开:5 | 0.9399 | 0.9369 | 0.9387 |
| NEZHA | 分开:0.7<br/>联合:0.3 | 是 | 100 | 30000 | 5 | 联合:10<br/>分开:5 | 0.9395 | 0.9367 | 0.9384 |
| NEZHA | 分开:0.7<br/>联合:0.3 | 是 | 100 | 30000 | 5 | 联合:10<br/>分开:5 | 0.9401 | 0.9347 | 0.9379 |
| NEZHA | 分开训练 | 是 | 100 | 30000 | 5 | 5 | 0.9369 | 0.9349 | 0.9361 |
| NEZHA | 预训练加入分组NSP任务<br>分开:0.7<br/>联合:0.3 | 是 | 100 | 30000 | 5 | 联合:10<br/>分开:5 | 0.9363 | 0.9318 | 0.9345 |
| BERT | 分开:0.7<br/>联合:0.3 | 是 | 100 | 30000 | 5 | 联合:10<br/>分开:5 | 0.9350 | 0.9288 | 0.9325 |
| NEZHA | 分开训练 | 否 | 100 | 30000 | 5 | 5 | 0.9326 | 0.9302 | 0.9316 |
| NEZHA | 分开训练 | 否 | 100 | 30000 | 10 | 10 | 0.9314 | 0.9264 | 0.9294 |
| NEZHA | 联合训练 | 否 | 100     | 100000 | 5   | 5  | 0.9257 | 0.9142 | 0.9212 |
| NEZHA | 联合训练 | 否 | 100 | 30000 | 5 | 5 | 0.9266 | 0.9099 | 0.9200 |

#### 3.1.2 DL

| 模型                               | 训练策略 | seq长度 | 训练fold | 训练epoch | 任务1得分 | 任务2得分 | 总得分 |
| ---------------------------------- | -------- | ------- | -------- | --------- | --------- | --------- | ------ |
| HAN:0.5<br/>DPCNN:0.3<br/>LSTM:0.2 | 分开训练 | 55      | 10       | 15        | -         | -         | 0.9306 |
| HAN                                | 分开训练 | 55      | 10       | 15        | -         | -         | 0.9231 |
| DPCNN                              | 分开训练 | 55      | 10       | 15        | -         | -         | 0.9202 |
| LSTM                               | 分开训练 | 55      | 10       | 15        | 0.9217    | 0.9160    | 0.9195 |
| LSTM                               | 分开训练 | 100     | 15       | 30        | 0.9205    | 0.9178    | 0.9194 |
| DPCNN                              | 联合训练 | 55      | 15       | 30        | 0.9255    | 0.9049    | 0.9173 |
| LSTM                               | 分开训练 | 70      | 5        | 10        | 0.9171    | 0.9030    | 0.9115 |
| LSTM                               | 联合训练 | 55      | 15       | 30        | 0.9181    | 0.9006    | 0.9111 |
| LSTM                               | 分开训练 | 55      | 8        | 15        | 0.9153    | 0.9019    | 0.9099 |

#### 3.1.3 融合

| 模型                 | NEZHA训练策略         | DL训练策略 | 余弦退火 | seq长度             | 预训练步数 | 训练fold | 训练epoch          | 任务1得分 | 任务2得分 | 总得分 |
| -------------------- | --------------------- | ---------- | -------- | ------------------- | ---------- | -------- | ------------------ | --------- | --------- | ------ |
| NEZHA:0.8<br/>DL:0.2 | 分开:0.7<br/>联合:0.3 | 分开训练   | 是       | NEZHA:100<br/>DL:55 | 30000      | 5        | 联合:10<br/>分开:5 | 0.9408    | 0.9383    | 0.9398 |
| NEZHA:0.7<br/>DL:0.3 | 分开:0.7<br/>联合:0.3 | 分开训练   | 是       | NEZHA:100<br/>DL:55 | 30000      | 5        | 联合:10<br/>分开:5 | 0.9397    | 0.9381    | 0.9391 |

### 3.2 B榜

| 模型                 | NEZHA训练策略         | DL训练策略 | 余弦退火 | seq长度             | 预训练步数 | NEZHA训练fold | NEZHA训练epoch     | DL训练fold | DL训练epoch | 任务1得分 | 任务2得分 | 总得分 |
| -------------------- | --------------------- | ---------- | -------- | ------------------- | ---------- | ------------- | ------------------ | ---------- | ----------- | --------- | --------- | ------ |
| NEZHA:0.8<br/>DL:0.2 | 分开:0.8<br/>联合:0.2 | 分开训练   | 是       | NEZHA:100<br>DL:55  | 30000      | 8             | 联合:10<br/>分开:5 | 10         | 15          | 0.9433    | 0.9412    | 0.9425 |
| NEZHA:0.8<br/>DL:0.2 | 分开:0.8<br/>联合:0.2 | 分开训练   | 是       | NEZHA:100<br/>DL:55 | 50000      | 8             | 联合:10<br/>分开:5 | 15         | 15          | 0.9433    | 0.9408    | 0.9423 |