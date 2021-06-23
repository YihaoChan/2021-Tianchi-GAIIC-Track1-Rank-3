import directory
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from helper.seed import seed
from run import RunConfig
import os
from helper.nezha_model import NeZhaForSequenceClassification
from transformers import BertModel

torch.cuda.empty_cache()


def region_test_pro():
    test_df = pd.read_csv(directory.SEMI_TEST_SET_A_PATH, header=None)

    test_df.columns = ['report_ID', 'description']

    test_df.drop(['report_ID'], axis=1, inplace=True)

    test_df['description'] = [i.strip('|').strip() for i in test_df['description'].values]

    test_num = len(test_df)

    for test_idx in tqdm(range(test_num)):
        des = test_df.loc[test_idx, 'description']

        des = [int(word) for word in des.split(' ')]

        test_df.loc[test_idx, 'description'] = des

    return test_df


def collect_type_path():
    """
    收集不同任务的权重路径名
    """
    all_weights_path = os.listdir(directory.MODEL_DIR)

    weights_path_list = []

    for weight_path in all_weights_path:
        method = weight_path.split('_')[0]  # 分开训练还是联合训练

        task_name = weight_path.split('_')[1]  # 训练的任务：区域/类型

        if method == 'separate' and task_name == 'region':
            weights_path_list.append(weight_path)

    return weights_path_list


def load_model(net, weight_path):
    model = net.to(run_config.device)

    model.load_state_dict(torch.load(weight_path))

    model.eval()

    return model


@torch.no_grad()
def predict(texts, net):
    """
    单一模型预测
    """
    single_pred = []

    for text in texts:
        # 截断填充
        text_len = len(text)

        if text_len > run_config.seq_len:
            text = text[:run_config.seq_len]
        else:
            text = text + [858] * (run_config.seq_len - text_len)

        text = torch.from_numpy(np.array(text))

        text = text.unsqueeze(0)

        text = text.type(torch.LongTensor).to(run_config.device)

        # 预测
        pred = net(text)

        pred = pred.sigmoid().detach().cpu().numpy()[0]

        single_pred.append(pred)

    return np.array(single_pred)


def submit(res):
    str_w = ''

    pred_num = len(res)

    with open(directory.REGION_RESULT_PATH, 'w') as f:
        for i in range(pred_num):
            pred = res[i]

            pred = [str(p) for p in pred]

            pred = ' '.join(pred)

            str_w += str(i) + '|,|' + pred + '\n'

        str_w = str_w.strip('\n')

        f.write(str_w)


def main():
    test_df = region_test_pro()

    test_des = test_df['description'].values

    weights_path_list = collect_type_path()

    pred_list = 0

    model = NeZhaForSequenceClassification(BertModel.from_pretrained(directory.PRETRAIN_DIR), 17)

    for weight_rel_path in tqdm(weights_path_list):
        weight_full_path = "%s/%s" % (directory.MODEL_DIR, weight_rel_path)

        model = load_model(model, weight_full_path)

        single_pred = predict(test_des, model)

        pred_list += single_pred

    res = (pred_list / len(weights_path_list)).tolist()

    submit(res)


if __name__ == '__main__':
    seed()

    run_config = RunConfig()

    main()
