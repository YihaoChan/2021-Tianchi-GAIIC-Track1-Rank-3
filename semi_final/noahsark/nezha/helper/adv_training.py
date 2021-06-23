from keras import backend as K
import numpy as np
from bert4keras.backend import keras
import tensorflow as tf
import os
import random

seed_value = 2021
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)


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
