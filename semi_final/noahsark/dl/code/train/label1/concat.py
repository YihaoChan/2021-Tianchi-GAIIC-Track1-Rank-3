import pandas as pd
import directory

# 对初赛和复赛的数据进行拼接
train_df1 = pd.read_csv(directory.PREM_TRAIN_SET_PATH, header=None)
train_df2 = pd.read_csv(directory.SEMI_TRAIN_SET_PATH, header=None)

train_df1.columns = ['report_ID', 'description', 'label']
train_df2.columns = ['report_ID', 'description', 'label', 'label2']

train_df1.drop(['report_ID'], axis=1, inplace=True)
train_df2.drop(['report_ID'], axis=1, inplace=True)
train_df2.drop(['label2'], axis=1, inplace=True)

train_dfconcat = pd.concat([train_df1, train_df2])
description = train_dfconcat['description'].values
label = train_dfconcat['label'].values

str_w = ''
with open('./dl/code/train/label1/concat.csv', 'w') as f:
    for i in range(0, 30000):
        str_w += str(i) + '|' + ',' + description[i] + ',' + label[i] + '\n'
    str_w = str_w.strip('\n')
    f.write(str_w)
