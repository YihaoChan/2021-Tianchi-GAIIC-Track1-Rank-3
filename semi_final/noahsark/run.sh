#bin/bash 

# NEZHA
python3 ./nezha/run.py

# DL
# 拼接初赛数据和复赛数据
python3 ./dl/code/train/label1/concat.py
# 运行LSTM
python3 ./dl/code/train/label1/lstm_train.py
python3 ./dl/code/test/label1/lstm_infer.py
python3 ./dl/code/train/label2/lstm_train.py
python3 ./dl/code/test/label2/lstm_infer.py
python3 ./dl/prediction_result/lstmMerge2label.py
# 运行DPCNN
python3 ./dl/code/train/label1/dpcnn_train.py
python3 ./dl/code/test/label1/dpcnn_infer.py
python3 ./dl/code/train/label2/dpcnn_train.py
python3 ./dl/code/test/label2/dpcnn_infer.py
python3 ./dl/prediction_result/dpcnnMerge2label.py
# 运行HAN
python3 ./dl/code/train/label1/han_train.py
python3 ./dl/code/test/label1/han_infer.py
python3 ./dl/code/train/label2/han_train.py
python3 ./dl/code/test/label2/han_infer.py
python3 ./dl/prediction_result/hanMerge2label.py
# 融合结果
python3 ./dl/code/test/cvs_fusion.py

# NEZHA + DL融合
python3 ./merge.py