import pandas as pd

submit_han_label1 = pd.read_csv('./dl/prediction_result/label1/submit_han.csv', header=None, dtype=str)
submit_han_label2 = pd.read_csv('./dl/prediction_result/label2/submit_han.csv', header=None, dtype=str)

submit_han_label1.columns = ['report_ID', 'prediction']
submit_han_label2.columns = ['report_ID', 'prediction']

new_han_label1 = [i.strip('|').strip() for i in submit_han_label1['prediction'].values]
new_han_label2 = [i.strip('|').strip() for i in submit_han_label2['prediction'].values]
submit_han_label1['prediction'] = new_han_label1
submit_han_label2['prediction'] = new_han_label2

submit_han = submit_han_label2
submit_han['prediction'] = submit_han_label1['prediction'].map(str) + ' ' + submit_han_label2['prediction'].map(str)

submit_hanID = submit_han['report_ID'].values
label_han = submit_han['prediction'].values

str_w = ''
with open('./dl/prediction_result/submit_han.csv', 'w') as f:
    for i in range(0, 5000):
        str_w += submit_hanID[i] + ',' + '|' + label_han[i] + '\n'
    str_w = str_w.strip('\n')
    f.write(str_w)
