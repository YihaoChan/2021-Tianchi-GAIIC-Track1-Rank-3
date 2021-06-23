import pandas as pd

submit_dpcnn_label1 = pd.read_csv('./dl/prediction_result/label1/submit_dpcnn.csv', header=None, dtype=str)
submit_dpcnn_label2 = pd.read_csv('./dl/prediction_result/label2/submit_dpcnn.csv', header=None, dtype=str)

submit_dpcnn_label1.columns = ['report_ID', 'prediction']
submit_dpcnn_label2.columns = ['report_ID', 'prediction']

new_dpcnn_label1 = [i.strip('|').strip() for i in submit_dpcnn_label1['prediction'].values]
new_dpcnn_label2 = [i.strip('|').strip() for i in submit_dpcnn_label2['prediction'].values]
submit_dpcnn_label1['prediction'] = new_dpcnn_label1
submit_dpcnn_label2['prediction'] = new_dpcnn_label2

submit_dpcnn = submit_dpcnn_label2
submit_dpcnn['prediction'] = submit_dpcnn_label1['prediction'].map(str) + ' ' + submit_dpcnn_label2['prediction'].map(
    str)

submit_dpcnnID = submit_dpcnn['report_ID'].values
label_dpcnn = submit_dpcnn['prediction'].values

str_w = ''
with open('./dl/prediction_result/submit_dpcnn.csv', 'w') as f:
    for i in range(0, 5000):
        str_w += submit_dpcnnID[i] + ',' + '|' + label_dpcnn[i] + '\n'
    str_w = str_w.strip('\n')
    f.write(str_w)
