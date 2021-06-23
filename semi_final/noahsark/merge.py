# coding: utf-8
import pandas as pd
import numpy as np


def main():
    nezha_result = pd.read_csv('./result_nezha.csv', header=None)
    nezha_result.columns = ['report_ID', 'label']

    dl_result = pd.read_csv('./result_dl.csv', header=None)
    dl_result.columns = ['report_ID', 'label']

    new_label_nezha = [i.strip('|').strip() for i in nezha_result['label'].values]
    nezha_result['label'] = new_label_nezha

    new_label_dl = [i.strip('|').strip() for i in dl_result['label'].values]
    dl_result['label'] = new_label_dl

    final_result = ['0' for _ in range(len(nezha_result))]
    prob = np.zeros(29)

    for i in range(0, len(new_label_nezha)):
        str2list_nezha = new_label_nezha[i].split()
        str2list_dl = new_label_dl[i].split()

        copy_nezha = str2list_nezha
        copy_dl = str2list_dl

        for j in range(0, len(str2list_nezha)):
            prob[j] = 0.8 * float(copy_nezha[j]) + \
                      0.2 * float(copy_dl[j])

        final_result[i] = ' '.join(str(i) for i in prob)

    sub_id = nezha_result['report_ID'].values
    str_w = ''

    with open('./result.csv', 'w') as f:
        for i in range(0, len(nezha_result)):
            str_w += sub_id[i] + ',' + '|' + final_result[i] + '\n'

        str_w = str_w.strip('\n')

        f.write(str_w)


if __name__ == '__main__':
    main()
