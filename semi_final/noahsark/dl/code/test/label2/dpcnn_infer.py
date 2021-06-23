from dpcnn import DPCNN
from config import DPCNNConfig
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import directory


def load_model(weight_path):
    print(weight_path)
    model = DPCNN(embed_num=859)
    model.load_state_dict(torch.load(weight_path))  # 返回的是一个OrderDict，存储了网络结构的名字和对应的参数
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(texts):
    pres_all = []
    for text in tqdm(texts):
        text = [int(i) for i in text.split(' ')]
        # 统一样本的长度，这里选择55个词语作为样本长度，多的截断，少的补齐(用858补齐)
        seq_len = DPCNNConfig.seq_len
        if len(text) > seq_len:
            text = text[:seq_len]
        else:
            text = text + [858] * (seq_len - len(text))

        text = torch.from_numpy(np.array(text))
        text = text.unsqueeze(0)
        text = text.type(torch.LongTensor).cuda()

        for i in range(len(model_list)):
            model = model_list[i]
            outputs = model(text)
            # print("outputs:",outputs)
            outputs = outputs.sigmoid().detach().cpu().numpy()
            # print("outputs:",outputs)
            if i == 0:
                pres_fold = outputs / len(model_list)
            else:
                pres_fold += outputs / len(model_list)

        # print("dpcnn_pres_fold:",pres_fold)
        pres_fold = [str(p) for p in pres_fold]
        pres_fold = ' '.join(pres_fold)
        pres_all.append(pres_fold)
    return pres_all


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_list = []
    n_splits = DPCNNConfig.n_splits

    for i in range(n_splits):
        model_list.append(load_model('./dl/user_data/model_data/label2/dpcnnfold_' + str(i + 1) + '_best.pth'))

    test_df = pd.read_csv(directory.SEMI_TEST_SET_B_PATH, header=None)

    test_df.columns = ['report_ID', 'description']
    submit = test_df.copy()
    print("test_df:{}".format(test_df.shape))
    new_des = [i.strip('|').strip() for i in test_df['description'].values]

    # 获取停用词
    stopwords_path = './dl/code/test/label2/stopwords.txt'
    stopwords = []
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line) > 0:
                stopwords.append(line.strip())

    # 去掉new_des_test中的停用词
    for j in range(0, len(new_des)):
        str2lst = new_des[j].split()
        copy = str2lst[:]
        for i in copy:
            if i in stopwords:
                copy.remove(i)
        str2lst = copy
        lst2str = " ".join(str(i) for i in str2lst)
        new_des[j] = lst2str

    test_df['description'] = new_des
    sub_id = test_df['report_ID'].values

    print(sub_id[0])

    save_dir = './dl/prediction_result/label2/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pres_all = predict(new_des)

    str_w = ''
    with open(save_dir + 'submit_dpcnn.csv', 'w') as f:
        for i in range(len(sub_id)):
            str_w += sub_id[i] + ',' + '|' + pres_all[i] + '\n'
        str_w = str_w.strip('\n')
        f.write(str_w)
