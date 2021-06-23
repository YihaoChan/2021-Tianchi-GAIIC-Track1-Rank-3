# -*- coding: utf-8 -*-
import pandas as pd
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from tqdm import tqdm
import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np
import os
import directory
from keras import backend as K
import tensorflow as tf
from run import RunConfig
import random
from helper.seed import seed
import warnings

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


class DataGenerate(DataGenerator):
    """
    数据生成器
    """

    def __iter__(self, random=False):
        dict_path = directory.VOCAB_PATH
        tokenizer = Tokenizer(dict_path, do_lower_case=True)

        batch_token_ids, batch_segment_ids, batch_labels = [], [], []

        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=run_config.seq_len)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(label)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)

                yield [batch_token_ids, batch_segment_ids], batch_labels

                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def create_classify_model(bert_config_path, bert_checkpoint_path):
    bert = build_transformer_model(
        config_path=bert_config_path,
        checkpoint_path=bert_checkpoint_path,
        model='bert',
        return_keras_model=False,
    )

    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)

    output = Dense(
        units=29,
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

        if method == 'joint':
            weights_path_list.append(weight_path)

    return weights_path_list


def submit(res, all_test, sub_id):
    pres_all = []

    for i in range(len(all_test)):
        line = [r / run_config.n_splits for r in list(np.sum([res[j][i] for j in range(len(res))], axis=0))]
        pres_all.append(line)

    pres_all = [' '.join(map(str, r)) for r in pres_all]

    str_w = ''

    with open(directory.JOINT_RESULT_PATH, 'w') as f:
        for i in range(len(pres_all)):
            str_w += sub_id[i] + ',' + '|' + pres_all[i] + '\n'

        str_w = str_w.strip('\n')

        f.write(str_w)


def main():
    seed()

    all_test, sub_id = read_test()
    all_test = np.array(all_test)
    all_test = DataGenerate([(' '.join([str(j) for j in all_test[i]]), []) for i in range(len(all_test))], 1)

    kfold_pre = []

    weights_path_list = collect_type_path()

    model = create_classify_model(
        directory.BERT_CONFIG_PATH, '%s/model.ckpt-%d' % (directory.CHECKPOINT_PATH, run_config.num_train_steps)
    )

    for weight_path in tqdm(weights_path_list):
        model.load_weights('%s/%s' % (directory.MODEL_DIR, weight_path))

        model_pre = []

        for test in all_test.__iter__():
            model_pre.append(list(model.predict(test[0]))[0])

        kfold_pre.append(model_pre)

    submit(kfold_pre, all_test, sub_id)


if __name__ == '__main__':
    run_config = RunConfig()

    main()
