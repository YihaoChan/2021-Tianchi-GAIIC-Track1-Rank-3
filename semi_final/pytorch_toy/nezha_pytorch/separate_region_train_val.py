from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from helper.nezha_model import NeZhaForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from helper.metric import metric
from helper.fgm_adv import FGM
from helper.seed import seed
from run import RunConfig
import torch
import torch.nn as nn
import numpy as np
import os
import warnings
import directory
import multiprocessing
from transformers import BertModel
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

torch.cuda.empty_cache()


def region_train_pro():
    train_prem_df = pd.read_csv(directory.PREM_TRAIN_SET_PATH, header=None)
    train_prem_df.columns = ['report_ID', 'description', 'region']
    train_semi_df = pd.read_csv(directory.SEMI_TRAIN_SET_PATH, header=None)
    train_semi_df.columns = ['report_ID', 'description', 'region', 'category']

    train_df = pd.concat([train_prem_df, train_semi_df], axis=0, ignore_index=True)

    train_df.drop(['report_ID', 'category'], axis=1, inplace=True)
    train_df['description'] = [i.strip('|').strip() for i in train_df['description'].values]
    train_df['region'] = [i.strip('|').strip() for i in train_df['region'].values]

    train_num = len(train_df)

    for train_idx in tqdm(range(train_num)):
        des = train_df.loc[train_idx, 'description']

        des = [int(word) for word in des.split(' ')]

        train_df.loc[train_idx, 'description'] = des

    return train_df


class RegionTextDataset(Dataset):
    def __init__(self, df, idx):
        super().__init__()
        self.df = df.loc[idx, :].reset_index(drop=True)

        self.description = df['description'].values

        self.labels = df['region'].values

    @staticmethod
    def get_dummy(classes):
        """
        标签转为0/1向量
        """
        label = [0] * 17

        if classes == '':
            return label
        else:
            temp = [int(i) for i in classes.strip().split(' ')]

            for i in temp:
                label[i] = 1

        return label

    @staticmethod
    def des_padding(des_list):
        """
        截断文本，少的用858填充，多的直接截断
        """
        des_len = len(des_list)

        if des_len > run_config.seq_len:
            des = des_list[:run_config.seq_len]
        else:
            des = des_list + [858] * (run_config.seq_len - des_len)

        return des

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        des = self.description[idx]

        label = self.labels[idx]

        padding_des = self.des_padding(des)

        label = self.get_dummy(label)

        return np.array(padding_des), np.array(label)


def train(model, train_loader, val_loader, fold):
    fold += 1

    best_metric = 1e-7
    best_epoch = 0

    iters = len(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=5e-4)

    criterion = torch.nn.BCEWithLogitsLoss()

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, last_epoch=-1)

    for epoch in range(run_config.separate_region_num_epochs):
        epoch += 1

        model.train(True)

        fgm = FGM(model)

        for batch_idx, (data, label) in enumerate(train_loader):
            batch_idx += 1

            data = data.type(torch.LongTensor).to(run_config.device)
            label = label.to(run_config.device).float()

            output = model(data).to(run_config.device)

            loss = criterion(output, label)

            optimizer.zero_grad()

            # 正常的grad
            loss.backward(retain_graph=True)

            # 对抗训练
            fgm.attack()
            loss_adv = criterion(output, label)
            loss_adv.backward(retain_graph=True)
            fgm.restore()

            # 梯度下降，更新参数
            optimizer.step()
            model.zero_grad()
            lr_scheduler.step()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

            print(
                '\rfold: {}, epoch: {}, batch: {} / {}, loss: {:.3f}'.format(
                    fold, epoch, batch_idx, iters, loss.item()
                ), end=''
            )

        val_metric = val(model, val_loader)

        print('\nval metric_loss: {:.4f}'.format(val_metric))

        best_model_out_path = "%s/separate_region_fold_%d_best.pth" % (directory.MODEL_DIR, fold)

        if val_metric > best_metric:
            best_metric = val_metric

            best_epoch = epoch

            torch.save(model.state_dict(), best_model_out_path)

            print("save best epoch: {}, best metric: {}".format(best_epoch, val_metric))

    print('fold: {}, best metric: {:.3f}, best epoch: {}'.format(fold, best_metric, best_epoch))

    return best_metric


@torch.no_grad()
def val(model, val_loader):
    model.eval()

    pred_list = []
    label_list = []

    for (data, label) in val_loader:
        data = data.type(torch.LongTensor).to(run_config.device)
        label = label.type(torch.LongTensor).to(run_config.device)

        output = model(data).to(run_config.device)

        pred_list += output.sigmoid().detach().cpu().numpy().tolist()

        label_list += label.detach().cpu().numpy().tolist()

    metric_loss = metric(label_list, pred_list)

    return metric_loss


def k_fold(train_df):
    folds = StratifiedKFold(n_splits=run_config.n_splits, shuffle=True, random_state=2021).split(
        np.arange(train_df.shape[0]), train_df.region.values
    )

    kfold_best = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        # 每一折都要产生一个新模型
        model = NeZhaForSequenceClassification(BertModel.from_pretrained(directory.PRETRAIN_DIR), 17)

        model = model.to(run_config.device)

        workers = multiprocessing.cpu_count()

        train_dataset = RegionTextDataset(train_df, train_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=run_config.batch_size, shuffle=True, num_workers=workers
        )

        val_dataset = RegionTextDataset(train_df, val_idx)

        val_loader = DataLoader(
            val_dataset, batch_size=run_config.batch_size, shuffle=False, num_workers=workers
        )

        best_loss = train(model, train_loader, val_loader, fold)

        kfold_best.append(best_loss)

    print("local cv:", kfold_best, np.mean(kfold_best))


def main():
    if not os.path.exists(directory.MODEL_DIR):
        os.makedirs(directory.MODEL_DIR)

    train_df = region_train_pro()

    k_fold(train_df)


if __name__ == '__main__':
    seed()

    run_config = RunConfig()

    main()
