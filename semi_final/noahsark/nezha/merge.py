# coding: utf-8
import pandas as pd
import numpy as np
import directory
from run import RunConfig


def main():
    separate_nezha = pd.read_csv(directory.SEPARATE_RESULT_PATH, header=None)
    separate_nezha.columns = ['report_ID', 'label']

    joint_nezha = pd.read_csv(directory.JOINT_RESULT_PATH, header=None)
    joint_nezha.columns = ['report_ID', 'label']

    new_label_separate_nezha = [i.strip('|').strip() for i in separate_nezha['label'].values]
    separate_nezha['label'] = new_label_separate_nezha

    new_label_joint_nezha = [i.strip('|').strip() for i in joint_nezha['label'].values]
    joint_nezha['label'] = new_label_joint_nezha

    final_result = ['0' for _ in range(len(separate_nezha))]
    prob = np.zeros(29)

    for i in range(0, len(new_label_separate_nezha)):
        str2list_separate_nezha = new_label_separate_nezha[i].split()
        str2list_joint_nezha = new_label_joint_nezha[i].split()

        copy_separate_nezha = str2list_separate_nezha
        copy_joint_nezha = str2list_joint_nezha

        for j in range(0, len(str2list_separate_nezha)):
            prob[j] = run_config.separate_weight * float(copy_separate_nezha[j]) + \
                      run_config.joint_weight * float(copy_joint_nezha[j])

        final_result[i] = ' '.join(str(i) for i in prob)

    sub_id = separate_nezha['report_ID'].values
    str_w = ''

    with open(directory.SUBMISSION_PATH, 'w') as f:
        for i in range(0, len(separate_nezha)):
            str_w += sub_id[i] + ',' + '|' + final_result[i] + '\n'

        str_w = str_w.strip('\n')

        f.write(str_w)


if __name__ == '__main__':
    run_config = RunConfig()

    main()
