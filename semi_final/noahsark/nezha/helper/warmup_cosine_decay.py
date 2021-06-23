import numpy as np
from bert4keras.backend import keras
from keras import backend as K
import tensorflow as tf
import os
import random

seed_value = 2021
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)


def cosine_decay_with_warmup(global_step, learning_rate_base, total_steps, warmup_learning_rate=0.0, warmup_steps=0,
                             hold_base_rate_steps=0):
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to warmup_steps.')

    learning_rate = 0.5 * learning_rate_base * (
            1 + np.cos(np.pi * (global_step - warmup_steps - hold_base_rate_steps)
                       / float(total_steps - warmup_steps - hold_base_rate_steps))
    )

    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)

    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to warmup_learning_rate.')

        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """
    继承Callback，实现对学习率的调度
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.learning_rate)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)

        K.set_value(self.model.optimizer.learning_rate, lr)

        if self.verbose > 0:
            print('\nBatch %05d: setting learning rate to %s.' % (self.global_step + 1, lr))
