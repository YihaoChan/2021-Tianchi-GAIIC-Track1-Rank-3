import pandas as pd

submit_lstm_label1 = pd.read_csv('./dl/prediction_result/label1/submit_lstm.csv', header=None, dtype=str)
submit_lstm_label2 = pd.read_csv('./dl/prediction_result/label2/submit_lstm.csv', header=None, dtype=str)

submit_lstm_label1.columns = ['report_ID', 'prediction']
submit_lstm_label2.columns = ['report_ID', 'prediction']

new_lstm_label1 = [i.strip('|').strip() for i in submit_lstm_label1['prediction'].values]
new_lstm_label2 = [i.strip('|').strip() for i in submit_lstm_label2['prediction'].values]
submit_lstm_label1['prediction'] = new_lstm_label1
submit_lstm_label2['prediction'] = new_lstm_label2

submit_lstm = submit_lstm_label2
submit_lstm['prediction'] = submit_lstm_label1['prediction'].map(str) + ' ' + submit_lstm_label2['prediction'].map(str)

submit_lstmID = submit_lstm['report_ID'].values
label_lstm = submit_lstm['prediction'].values
str_w = ''

with open('./dl/prediction_result/submit_lstm.csv', 'w') as f:
    for i in range(0, 5000):
        str_w += submit_lstmID[i] + ',' + '|' + label_lstm[i] + '\n'
    str_w = str_w.strip('\n')
    f.write(str_w)
