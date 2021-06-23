# -*- coding: utf-8 -*-
import pandas as pd
from keras.layers import Lambda, Dense
from tqdm import tqdm
import keras
from bert4keras.models import build_transformer_model
import numpy as np
import os
import directory
from keras import backend as K
import tensorflow as tf
from run import RunConfig
import random
import warnings
from helper.data_generator import DataGenerate

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

seed_value = 2021
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)


def read_test():
    test_df = pd.read_csv(directory.SEMI_TEST_SET_A_PATH, header=None)

    test_df.columns = ['report_ID', 'description']

    new_des = [map(int, i.strip('|').strip().split()) for i in test_df['description'].values]

    return [list(r) for r in new_des], test_df['report_ID']


def create_classify_model(bert_config_path, bert_checkpoint_path):
    bert = build_transformer_model(
        config_path=bert_config_path,
        checkpoint_path=bert_checkpoint_path,
        model='bert',
        return_keras_model=False,
    )

    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)

    output = Dense(
        units=17,
        activation='sigmoid',
        kernel_initializer=bert.initializer
    )(output)

    net = keras.models.Model(bert.model.input, output)

    return net


def collect_type_path():
    """
    收集不同任务的权重路径名
    """
    all_weights_path = os.listdir(directory.MODEL_DIR)

    weights_path_list = []

    for weight_path in all_weights_path:
        method = weight_path.split('_')[0]  # 分开训练还是联合训练

        task_name = weight_path.split('_')[1]  # 训练的任务：区域/类型

        if method == 'separate' and task_name == 'region':
            weights_path_list.append(weight_path)

    return weights_path_list


def region_task_predict(all_test, sub_id):
    all_test = np.array(all_test)
    all_test = DataGenerate([(' '.join([str(j) for j in all_test[i]]), []) for i in range(len(all_test))], 1)

    region_task_pre = []

    model = create_classify_model(
        directory.BERT_CONFIG_PATH,
        '%s/model.ckpt-%d' % (directory.CHECKPOINT_PATH, run_config.num_train_steps)
    )

    weights_path_list = collect_type_path()

    for weight_rel_path in tqdm(weights_path_list):
        model.load_weights('%s/%s' % (directory.MODEL_DIR, weight_rel_path))

        model_pre = []

        for test in all_test.__iter__():
            model_pre.append(list(model.predict(test[0]))[0])

        region_task_pre.append(model_pre)

    return region_task_pre


def write_region_result(single_task_res, all_test, sub_id):
    """
    生成单个任务的预测文件
    """
    pres_all = []

    for i in range(len(all_test)):
        line = [r / run_config.n_splits for r in
                list(np.sum([single_task_res[j][i] for j in range(len(single_task_res))], axis=0))]
        pres_all.append(line)

    pres_all = [' '.join(map(str, r)) for r in pres_all]

    str_w = ''

    with open(directory.REGION_RESULT_PATH, 'w') as f:
        for i in range(len(pres_all)):
            str_w += sub_id[i] + ',' + '|' + pres_all[i] + '\n'

        str_w = str_w.strip('\n')

        f.write(str_w)


def main():
    all_test, sub_id = read_test()

    category_res = region_task_predict(all_test, sub_id)
    write_region_result(category_res, all_test, sub_id)


if __name__ == '__main__':
    run_config = RunConfig()

    main()
