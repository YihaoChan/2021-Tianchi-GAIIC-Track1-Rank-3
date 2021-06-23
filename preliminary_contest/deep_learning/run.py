import torch
import os


class RunConfig(object):
    def __init__(self):
        self.device = torch.device('cuda')

        self.num_epochs = 25

        self.train_batch_size = 32

        self.val_batch_size = 64

        self.seq_len = 100

        self.n_splits = 20


def run():
    os.system('python3 ./train_val.py rcnn')
    os.system('python3 ./train_val.py rcnnattn')
    os.system('python3 ./train_val.py dpcnn')
    os.system('python3 ./predict.py')


if __name__ == '__main__':
    run()
