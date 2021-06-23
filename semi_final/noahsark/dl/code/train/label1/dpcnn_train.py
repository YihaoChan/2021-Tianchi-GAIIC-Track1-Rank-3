import pandas as pd
import numpy as np
from sklearn import metrics
import os
from sklearn.model_selection import StratifiedKFold
from dpcnn import DPCNN
from dpcnn_datasets import TextDataset
from config import DPCNNConfig
import torch
from torch.utils.data import DataLoader
import time
import warnings
import random
from seed import seed
import directory

warnings.filterwarnings("ignore")

random.seed(2021)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train_model(model, criterion, optimizer, lr_scheduler=None):
    total_iters = len(trainloader)
    # print('total_iters:{}'.format(total_iters))
    # since = time.time()
    best_loss = 1e7
    best_epoch = 0

    iters = len(trainloader)
    for epoch in range(1, max_epoch + 1):
        model.train(True)

        # begin_time = time.time()
        # print('learning rate:{}'.format(optimizer.param_groups[-1]['lr']))
        # print('Fold{} Epoch {}/{}'.format(fold + 1, epoch, max_epoch))
        # print('-' * 10)
        running_corrects_linear = 0
        count = 0
        train_loss = []

        for i, (inputs, labels) in enumerate(trainloader):
            count += 1
            inputs = inputs.type(torch.LongTensor).to(device)
            # print("inputs.shape:",inputs.shape)
            labels = labels.to(device).float()

            out_linear = model(inputs)
            loss = criterion(out_linear, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新cosine学习率
            if lr_scheduler != None:
                lr_scheduler.step(epoch + count / iters)

            train_loss.append(loss.item())

        val_loss = val_model(model, criterion)
        # print('valLogLoss: {:.4f} '.format(val_loss))

        best_model_out_path = model_save_dir + "/" + 'dpcnnfold_' + str(fold + 1) + '_best' + '.pth'

        # save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_out_path)
            # print("save best epoch: {}  best logloss: {}".format(best_epoch, val_loss))

    # print('Fold{} Best logloss: {:.3f} Best epoch:{}'.format(fold + 1, best_loss, best_epoch))
    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return best_loss


@torch.no_grad()
def val_model(model, criterion):
    dset_sizes = len(val_dataset)
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    pres_list = []
    labels_list = []

    for data in val_loader:
        inputs, labels = data
        inputs = inputs.type(torch.LongTensor)
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)

        pres_list += outputs.sigmoid().detach().cpu().numpy().tolist()
        labels_list += labels.detach().cpu().numpy().tolist()

    log_loss = metrics.log_loss(labels_list, pres_list) / 17.0

    return log_loss


if __name__ == "__main__":
    # 计时
    since = time.time()
    print('dpcnn_label1 starts')
    print('-' * 10)

    seed()

    # 处理训练数据集数据，显示训练数据集和测试数据集的大小
    train_df = pd.read_csv('./dl/code/train/label1/concat.csv', header=None)
    test_df = pd.read_csv(directory.SEMI_TEST_SET_B_PATH, header=None)
    train_df.columns = ['report_ID', 'description', 'label']
    test_df.columns = ['report_ID', 'description']
    train_df.drop(['report_ID'], axis=1, inplace=True)
    test_df.drop(['report_ID'], axis=1, inplace=True)
    print("train_df:{},test_df:{}".format(train_df.shape, test_df.shape))

    new_des = [i.strip('|').strip() for i in train_df['description'].values]

    # 去掉new_des中的停用词
    stopwords_path = './dl/code/train/label1/stopwords.txt'
    stopwords = []
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line) > 0:
                stopwords.append(line.strip())
    for j in range(0, len(new_des)):
        str2lst = new_des[j].split()
        copy = str2lst[:]
        for i in copy:
            if i in stopwords:
                copy.remove(i)
        str2lst = copy
        lst2str = " ".join(str(i) for i in str2lst)
        new_des[j] = lst2str

    new_label = [i.strip('|').strip() for i in train_df['label'].values]
    train_df['description'] = new_des
    train_df['label'] = new_label

    # 输出训练数据里面包含的正常样本数，这里的正常样本标签就使用[0,0,0,....,0,0]编码
    print('无异常样本:', train_df[train_df['label'] == ''].shape[0])  # 2622

    model_save_dir = './dl/user_data/model_data/label1'

    train_batch_size = DPCNNConfig.train_batch_size
    val_batch_size = DPCNNConfig.val_batch_size
    max_epoch = DPCNNConfig.max_epoch
    embed_num = 859
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.BCEWithLogitsLoss()  # 在Pytorch中，BCELoss和BCEWithLogitsLoss是一组常用的二元交叉熵损失函数，常用于二分类问题。

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    folds = StratifiedKFold(n_splits=DPCNNConfig.n_splits, shuffle=True, random_state=2021).split(
        np.arange(train_df.shape[0]), train_df.label.values
    )

    kfold_best = []

    for fold, (trn_idx, val_idx) in enumerate(folds):
        # print('train fold {}'.format(fold + 1))
        model = DPCNN(embed_num)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=DPCNNConfig.lr, weight_decay=5e-4)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5,
                                                                            last_epoch=-1)
        train_dataset = TextDataset(train_df, trn_idx)
        trainloader = DataLoader(train_dataset,
                                 batch_size=train_batch_size,
                                 shuffle=True,
                                 num_workers=0)
        val_dataset = TextDataset(train_df, val_idx)
        val_loader = DataLoader(val_dataset,
                                batch_size=val_batch_size,
                                shuffle=False,
                                num_workers=4)
        best_loss = train_model(model, criterion, optimizer, lr_scheduler=lr_scheduler)
        kfold_best.append(best_loss)

    print("local cv:", kfold_best, np.mean(kfold_best))
    print('-' * 10)

    # 计时，输出总时间
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
