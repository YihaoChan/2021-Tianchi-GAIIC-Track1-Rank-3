# -*- coding: utf-8 -*-
import pandas as pd
from bert4keras.snippets import sequence_padding, DataGenerator
from run import RunConfig
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

K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def read_test():
    test_df = pd.read_csv(directory.TEST_SET_B_PATH, header=None)  # 对B榜测试集进行预测

    test_df.columns = ['report_ID', 'description']

    new_des = [map(int, i.strip('|').strip().split()) for i in test_df['description'].values]

    return [list(r) for r in new_des], test_df['report_ID']


class DataGenerate(DataGenerator):
    """
    数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=seq_len)

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
        model='NEZHA',
        return_keras_model=False,
    )

    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)

    output = Dense(
        units=17,
        activation='sigmoid',
        kernel_initializer=bert.initializer
    )(output)

    net = keras.models.Model(bert.model.input, output)

    net.summary()  # 输出模型结构和参数数量

    return net


def submit(res):
    pres_all = []

    for i in range(len(all_test)):
        line = [r / n_splits for r in list(np.sum([res[j][i] for j in range(len(res))], axis=0))]
        pres_all.append(line)

    pres_all = [' '.join(map(str, r)) for r in pres_all]

    if not os.path.exists(directory.SUBMISSION_DIR):
        os.makedirs(directory.SUBMISSION_DIR)

    str_w = ''

    with open(directory.SUBMISSION_PATH, 'w') as f:
        for i in range(len(pres_all)):
            str_w += sub_id[i] + ',' + '|' + pres_all[i] + '\n'

        str_w = str_w.strip('\n')

        f.write(str_w)


if __name__ == '__main__':
    run_config = RunConfig()

    seq_len = run_config.seq_len
    n_splits = run_config.n_splits

    all_test, sub_id = read_test()
    all_test = np.array(all_test)
    all_test = DataGenerate([(' '.join([str(j) for j in all_test[i]]), []) for i in range(len(all_test))], 1)

    config_path = directory.BERT_CONFIG_PATH
    checkpoint_path = directory.CHECKPOINT_PATH + '/model.ckpt-30000'
    dict_path = directory.VOCAB_PATH

    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    kfold_pre = []

    all_weights = os.listdir(directory.MODEL_DIR)

    model = create_classify_model(config_path, checkpoint_path)

    for weight_rel_path in tqdm(all_weights):
        model.load_weights('%s/%s' %(directory.MODEL_DIR, weight_rel_path))

        model_pre = []

        for test in all_test.__iter__():
            model_pre.append(list(model.predict(test[0]))[0])

        kfold_pre.append(model_pre)

    submit(kfold_pre)
