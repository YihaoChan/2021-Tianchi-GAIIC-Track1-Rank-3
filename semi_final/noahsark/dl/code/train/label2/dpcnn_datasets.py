from torch.utils.data import Dataset
from config import DPCNNConfig
import numpy as np


class TextDataset(Dataset):

    def __init__(self, df, idx):

        super().__init__()

        df = df.loc[idx, :].reset_index(drop=True)

        self.text_lists = df['description'].values

        self.labels = df['label'].values

    def get_dumm(self, s):

        re = [0] * 12

        if s == '':

            return re

        else:

            tmp = [int(i) for i in s.strip().split(' ')]

            for i in tmp:
                re[i] = 1

        return re

    # __len__函数的作用是返回数据集的长度
    def __len__(self):

        return self.labels.shape[0]

    # __getitem__函数的作用是根据索引index遍历数据，一般返回image的Tensor形式和对应标注。当然也可以多返回一些其它信息，这个根据需求而定。
    # 在此统一样本的长度，这里选择55个词语作为样本长度，多的截断，少的补齐(用858补齐)
    def __getitem__(self, idx):

        text = self.text_lists[idx]

        text = [int(i) for i in text.split(' ')]

        seq_len = DPCNNConfig.seq_len

        if len(text) > seq_len:

            text = text[: seq_len]

        else:

            text = text + [858] * (seq_len - len(text))

        label = self.labels[idx]

        # print(label,[i for i in label])

        label = self.get_dumm(label)

        return np.array(text), np.array(label)
