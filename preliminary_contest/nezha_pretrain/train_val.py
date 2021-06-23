# -*- coding: utf-8 -*-
from tqdm import tqdm
import pandas as pd
import os
from bert4keras.tokenizers import Tokenizer
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
import keras
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import activations
from keras.layers import Layer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, Bidirectional
import tensorflow as tf
from keras.callbacks import EarlyStopping
import numpy as np
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense
from keras import backend as K
from seed import seed
from run import RunConfig
import warnings
import directory

warnings.filterwarnings('ignore')

K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def adversarial_training(net, embedding_name, epsilon=1.0):
    """
    对抗训练
    model: 需要添加对抗训练的keras模型，
    embedding_name: model里边Embedding层的名字。要在模型compile之后使用.
    """
    if net.train_function is None:  # 如果还没有训练函数
        net._make_train_function()  # 手动make

    old_train_function = net.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in net.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break

    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(net.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    inputs_layer = (net._feed_inputs +
                    net._feed_targets +
                    net._feed_sample_weights)  # 所有输入层

    embedding_gradients = K.function(
        inputs=inputs_layer,
        outputs=[gradients],
        name='embedding_gradients',
    )

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度

        delta = epsilon * grads / (np.sqrt((grads ** 2).sum()) + 1e-8)  # 计算扰动

        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动

        outputs = old_train_function(inputs)  # 梯度下降

        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动

        return outputs

    net.train_function = train_function  # 覆盖原训练函数


def search_layer(inputs, name, exclude=None):
    """
    根据inputs和name来搜索层
    inputs为某个层或某个层的输出；name为目标层的名字
    根据inputs一直往上递归搜索，直到发现名字为name的层为止；如果找不到，那就返回None
    """
    if exclude is None:
        exclude = set()

    if isinstance(inputs, keras.layers.Layer):
        layer = inputs
    else:
        layer = inputs._keras_history[0]

    if layer.name == name:
        return layer
    elif layer in exclude:
        return None
    else:
        exclude.add(layer)

        inbound_layers = layer._inbound_nodes[0].inbound_layers

        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]

        if len(inbound_layers) > 0:
            for layer in inbound_layers:
                layer = search_layer(layer, name, exclude)
                if layer is not None:
                    return layer


def read_train():
    train_df = pd.read_csv(directory.TRAIN_SET_PATH, header=None)

    train_df.columns = ['report_ID', 'description', 'label']
    train_df.drop(['report_ID'], axis=1, inplace=True)

    new_des = [map(int, i.strip('|').strip().split()) for i in train_df['description'].values]

    label = []

    for i in train_df['label'].values:
        i = i.strip('|').strip()

        line = [0 for _ in range(17)]

        if not i:
            label.append(line)
            continue

        for t in i.split():
            line[int(t)] = 1

        label.append(line)

    return [list(r) for r in new_des], label


def auc(y_true, y_pred):
    auc_score = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc_score


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


if __name__ == '__main__':
    seed()

    run_config = RunConfig()

    all_datas, all_labels = read_train()

    X = np.array(all_datas)
    Y = np.array(all_labels)

    seq_len = run_config.seq_len
    batch_size = run_config.batch_size
    n_splits = run_config.n_splits
    num_epochs = run_config.num_epochs

    count = len(all_labels)
    cvscores = 0

    config_path = directory.BERT_CONFIG_PATH
    checkpoint_path = directory.CHECKPOINT_PATH + '/model.ckpt-30000'
    dict_path = directory.VOCAB_PATH

    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    k_fold = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2021)

    fold = 1

    if not os.path.exists(directory.MODEL_DIR):
        os.makedirs(directory.MODEL_DIR)

    for train_index, valid_index in k_fold.split(X, Y):
        print("fold: %d/%d" % (fold, n_splits))

        X_train, label_train = X[train_index], Y[train_index]
        X_valid, label_valid = X[valid_index], Y[valid_index]

        x_train = [(' '.join([str(j) for j in X_train[i]]), list(label_train[i])) for i in range(len(X_train))]
        x_valid = [(' '.join([str(j) for j in X_valid[i]]), list(label_valid[i])) for i in range(len(X_valid))]

        train_D = DataGenerate(x_train, batch_size)
        valid_D = DataGenerate(x_valid, batch_size)

        model = create_classify_model(config_path, checkpoint_path)
        model.compile(loss='binary_crossentropy', optimizer=Adam(1e-5))

        adversarial_training(model, 'Embedding-Token', 0.1)

        model.fit(
            train_D.forfit(),
            steps_per_epoch=len(train_D),
            epochs=num_epochs,
            validation_data=valid_D.forfit(),
            validation_steps=len(valid_D)
        )

        model.save_weights(directory.MODEL_DIR + '/fold_%s_weights.h5' % fold)

        fold += 1
