import os
import random
import tensorflow as tf
import numpy as np


def seed():
    seed_value = 2021
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.set_random_seed(seed_value)
    tf.compat.v1.set_random_seed(seed_value)
