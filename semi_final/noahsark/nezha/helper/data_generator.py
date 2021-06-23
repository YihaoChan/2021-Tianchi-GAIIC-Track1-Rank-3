import directory
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
from run import RunConfig
import tensorflow as tf
import os
import random
import numpy as np

seed_value = 2021
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)


class DataGenerate(DataGenerator):
    """
    数据生成器
    """

    def __iter__(self, random=False):
        run_config = RunConfig()

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
