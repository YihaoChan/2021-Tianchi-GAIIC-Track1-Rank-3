# -*- coding: utf-8 -*-
import pandas as pd
import os
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import tensorflow as tf
import numpy as np
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from keras.layers import Lambda, Dense
from keras import backend as K
from helper.seed import seed
from run import RunConfig
import warnings
import directory
import random
from helper.warmup_cosine_decay import WarmUpCosineDecayScheduler
from helper.data_generator import DataGenerate
from helper.adv_training import adversarial_training

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


def read_train():
    train_df = pd.read_csv(directory.SEMI_TRAIN_SET_PATH, header=None)
    train_df.columns = ['report_ID', 'description', 'region', 'category']
    train_df.drop(['report_ID'], axis=1, inplace=True)
    train_df = train_df.fillna(value={'category': -1})

    new_des = [map(int, i.strip('|').strip().split()) for i in train_df['description'].values]

    label = []

    train_num = len(train_df)

    for i in range(train_num):
        regions = train_df.loc[i, 'region']
        categories = train_df.loc[i, 'category']

        regions = regions.strip('|').strip()

        region_onehot = [0 for _ in range(17)]
        category_onehot = [0 for _ in range(12)]

        if regions != '':
            for region in regions.split():
                region_onehot[int(region)] = 1

        if categories != -1:
            for category in categories.split():
                category_onehot[int(category)] = 1

        label.append(region_onehot + category_onehot)

    return [list(r) for r in new_des], label


def create_classify_model(bert_config_path, bert_checkpoint_path):
    bert = build_transformer_model(
        config_path=bert_config_path,
        checkpoint_path=bert_checkpoint_path,
        model='NEZHA',
        return_keras_model=False,
    )

    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)

    output = Dense(
        units=29,
        activation='sigmoid',
        kernel_initializer=bert.initializer,
    )(output)

    net = keras.models.Model(bert.model.input, output)

    return net


def main():
    seed()

    all_datas, all_labels = read_train()

    X = np.array(all_datas)
    Y = np.array(all_labels)

    k_fold = MultilabelStratifiedKFold(n_splits=run_config.n_splits, shuffle=True, random_state=2021)

    fold = 1

    if not os.path.exists(directory.MODEL_DIR):
        os.makedirs(directory.MODEL_DIR)

    for train_index, valid_index in k_fold.split(X, Y):
        print("Fold: %d/%d" % (fold, run_config.n_splits))

        X_train, label_train = X[train_index], Y[train_index]
        X_valid, label_valid = X[valid_index], Y[valid_index]

        x_train = [(' '.join([str(j) for j in X_train[i]]), list(label_train[i])) for i in range(len(X_train))]
        x_valid = [(' '.join([str(j) for j in X_valid[i]]), list(label_valid[i])) for i in range(len(X_valid))]

        train_D = DataGenerate(x_train, run_config.batch_size)
        valid_D = DataGenerate(x_valid, run_config.batch_size)

        model = create_classify_model(
            directory.BERT_CONFIG_PATH, '%s/model.ckpt-%d' % (directory.CHECKPOINT_PATH, run_config.num_train_steps)
        )

        optimizer = keras.optimizers.Adam(learning_rate=1e-5)
        model.compile(loss='binary_crossentropy', optimizer=optimizer)

        adversarial_training(model, 'Embedding-Token', 0.1)

        # Warm up + 余弦退火
        sample_count = 30000
        warmup_epoch = 2
        learning_rate_base = 0.0001
        total_steps = int(run_config.joint_num_epochs * sample_count / run_config.batch_size)
        warmup_steps = int(warmup_epoch * sample_count / run_config.batch_size)
        warm_up_lr = WarmUpCosineDecayScheduler(
            learning_rate_base=learning_rate_base,
            total_steps=total_steps,
            warmup_learning_rate=4e-06,
            warmup_steps=warmup_steps,
            hold_base_rate_steps=5
        )

        model.fit(
            train_D.forfit(),
            steps_per_epoch=len(train_D),
            epochs=run_config.joint_num_epochs,
            validation_data=valid_D.forfit(),
            validation_steps=len(valid_D),
            callbacks=[warm_up_lr],
            verbose=2
        )

        model.save_weights('%s/joint_fold_%s_weights.h5' % (directory.MODEL_DIR, fold))

        fold += 1


if __name__ == '__main__':
    run_config = RunConfig()

    main()
