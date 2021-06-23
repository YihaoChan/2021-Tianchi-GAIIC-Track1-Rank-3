# coding: utf-8
import pandas as pd
import numpy as np
import directory
from run import RunConfig


def main():
    separate_bert = pd.read_csv(directory.SEPARATE_RESULT_PATH, header=None)
    separate_bert.columns = ['report_ID', 'label']

    joint_bert = pd.read_csv(directory.JOINT_RESULT_PATH, header=None)
    joint_bert.columns = ['report_ID', 'label']

    new_label_separate_bert = [i.strip('|').strip() for i in separate_bert['label'].values]
    separate_bert['label'] = new_label_separate_bert

    new_label_joint_bert = [i.strip('|').strip() for i in joint_bert['label'].values]
    joint_bert['label'] = new_label_joint_bert

    final_result = ['0' for _ in range(len(separate_bert))]
    prob = np.zeros(29)

    for i in range(0, len(new_label_separate_bert)):
        str2list_separate_bert = new_label_separate_bert[i].split()
        str2list_joint_bert = new_label_joint_bert[i].split()

        copy_separate_bert = str2list_separate_bert
        copy_joint_bert = str2list_joint_bert

        for j in range(0, len(str2list_separate_bert)):
            prob[j] = run_config.separate_weight * float(copy_separate_bert[j]) + \
                      run_config.joint_weight * float(copy_joint_bert[j])

        final_result[i] = ' '.join(str(i) for i in prob)

    sub_id = separate_bert['report_ID'].values
    str_w = ''

    with open(directory.SUBMISSION_PATH, 'w') as f:
        for i in range(0, len(separate_bert)):
            str_w += sub_id[i] + ',' + '|' + final_result[i] + '\n'

        str_w = str_w.strip('\n')

        f.write(str_w)


if __name__ == '__main__':
    run_config = RunConfig()

    main()
