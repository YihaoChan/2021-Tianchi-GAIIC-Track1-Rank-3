from load_net import gen_net
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from helper.metric import metric
from helper.preprocess import train_pro
from helper.dataset import TextDataset
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
import sys

warnings.filterwarnings("ignore")


def train(model, train_loader, val_loader, fold, run_config, model_config, net_name):
    fold += 1

    best_metric = 1e-7
    best_epoch = 0

    iters = len(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate, weight_decay=5e-4)

    criterion = torch.nn.BCEWithLogitsLoss()

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, last_epoch=-1)

    for epoch in range(run_config.num_epochs):
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

        val_metric = val(model, val_loader, run_config)

        print('\nval metric_loss: {:.4f}'.format(val_metric))

        best_model_out_path = "%s/%s_fold_%d_best.pth" % (directory.MODEL_DIR, net_name, fold)

        if val_metric > best_metric:
            best_metric = val_metric

            best_epoch = epoch

            torch.save(model.state_dict(), best_model_out_path)

            print("save best epoch: {}, best metric: {}".format(best_epoch, val_metric))

    print('fold: {}, best metric: {:.3f}, best epoch: {}'.format(fold, best_metric, best_epoch))

    return best_metric


@torch.no_grad()
def val(model, val_loader, run_config):
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


def k_fold(train_df, net_name, run_config):
    folds = StratifiedKFold(n_splits=run_config.n_splits, shuffle=True, random_state=7).split(
        np.arange(train_df.shape[0]), train_df.region.values
    )

    kfold_best = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        # 每一折都要产生一个新模型
        net, _, model_config = gen_net(net_name)

        model = net.to(run_config.device)

        workers = multiprocessing.cpu_count()

        train_dataset = TextDataset(train_df, train_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=run_config.train_batch_size, shuffle=True, num_workers=workers, pin_memory=True
        )

        val_dataset = TextDataset(train_df, val_idx)

        val_loader = DataLoader(
            val_dataset, batch_size=run_config.val_batch_size, shuffle=False, num_workers=workers, pin_memory=True
        )

        best_loss = train(model, train_loader, val_loader, fold, run_config, model_config, net_name)

        kfold_best.append(best_loss)

    print("local cv:", kfold_best, np.mean(kfold_best))


def main():
    net_name = sys.argv[1]

    if net_name not in ['rcnn', 'dpcnn', 'rcnnattn']:
        raise ValueError('Net_name should among rcnn, dpcnn and rcnnattn')

    if not os.path.exists(directory.MODEL_DIR):
        os.makedirs(directory.MODEL_DIR)

    seed()

    train_df = train_pro(directory.TRAIN_SET_PATH)

    k_fold(train_df, net_name, RunConfig())


if __name__ == '__main__':
    main()
