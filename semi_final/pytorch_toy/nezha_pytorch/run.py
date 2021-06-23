import os
import torch


class RunConfig(object):
    def __init__(self):
        self.device = torch.device('cuda:0')

        # 预训练轮数
        self.num_pretrain_epochs = 100

        # 预训练、训练的batch_size
        self.batch_size = 16

        # 分开训练：任务1(区域)的epoch
        self.separate_region_num_epochs = 10

        # 分开训练：任务2(类型)的epoch
        self.separate_category_num_epochs = 10

        self.seq_len = 100

        self.n_splits = 5


def run():
    os.system('python3 ./nezha_pytorch/corpus_vocab.py')
    os.system('python3 ./nezha_pytorch/pretraining.py')
    os.system('python3 ./nezha_pytorch/separate_region_train_val.py')
    os.system('python3 ./nezha_pytorch/separate_region_predict.py')
    os.system('python3 ./nezha_pytorch/separate_category_train_val.py')
    os.system('python3 ./nezha_pytorch/separate_category_predict.py')
    os.system('python3 ./nezha_pytorch/separate_submit.py')


if __name__ == '__main__':
    run()
