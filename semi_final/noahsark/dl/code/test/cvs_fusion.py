import pandas as pd
import numpy as np

submit_lstm = pd.read_csv('./dl/prediction_result/submit_lstm.csv', header=None)
submit_lstm.columns = ['report_ID', 'label']

submit_dpcnn = pd.read_csv('./dl/prediction_result/submit_dpcnn.csv', header=None)
submit_dpcnn.columns = ['report_ID', 'label']

submit_han = pd.read_csv('./dl/prediction_result/submit_han.csv', header=None)
submit_han.columns = ['report_ID', 'label']

print("submit_lstm:{}".format(submit_lstm.shape))
new_label_lstm = [i.strip('|').strip() for i in submit_lstm['label'].values]
submit_lstm['label'] = new_label_lstm

print("submit_dpcnn:{}".format(submit_dpcnn.shape))
new_label_dpcnn = [i.strip('|').strip() for i in submit_dpcnn['label'].values]
submit_dpcnn['label'] = new_label_dpcnn

print("submit_han:{}".format(submit_han.shape))
new_label_han = [i.strip('|').strip() for i in submit_han['label'].values]
submit_han['label'] = new_label_han

data_fusion = ['0' for i in range(5000)]
fusion = np.zeros(29)
for i in range(0, len(new_label_han)):  #
    str2lst_lstm = new_label_lstm[i].split()
    str2lst_dpcnn = new_label_dpcnn[i].split()
    str2lst_han = new_label_han[i].split()

    copy_lstm = str2lst_lstm
    copy_dpcnn = str2lst_dpcnn
    copy_han = str2lst_han

    for j in range(0, len(str2lst_han)):
        fusion[j] = 0.2 * float(copy_lstm[j]) + 0.3 * float(copy_dpcnn[j]) + 0.5 * float(copy_han[j])

    lst2str_fusion = " ".join(str(i) for i in fusion)
    data_fusion[i] = lst2str_fusion

subFusion_id = submit_han['report_ID'].values

str_w = ''
with open('./result_dl.csv', 'w') as f:
    for i in range(0, 5000):
        str_w += subFusion_id[i] + ',' + '|' + data_fusion[i] + '\n'
    str_w = str_w.strip('\n')
    f.write(str_w)
