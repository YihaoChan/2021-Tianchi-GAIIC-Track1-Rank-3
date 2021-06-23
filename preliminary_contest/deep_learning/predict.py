from load_net import gen_net
from helper.preprocess import test_pro
from tqdm import tqdm
import numpy as np
import torch
import directory
import os


def load_model(net, weight_path, run_config):
    model = net.to(run_config.device)

    model.load_state_dict(torch.load(weight_path))

    model.eval()

    return model


@torch.no_grad()
def predict(texts, net, run_config):
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

    with open(directory.SUBMISSION_PATH, 'w') as f:
        for i in range(pred_num):
            pred = res[i]

            pred = [str(p) for p in pred]

            pred = ' '.join(pred)

            str_w += str(i) + '|,|' + pred + '\n'

        str_w = str_w.strip('\n')

        f.write(str_w)


def main():
    if not os.path.exists(directory.SUBMISSION_DIR):
        os.makedirs(directory.SUBMISSION_DIR)

    test_df = test_pro(directory.TEST_SET_B_PATH)  # 对B榜测试集进行预测

    test_des = test_df['description'].values

    all_weights = os.listdir(directory.MODEL_DIR)

    pred_list = 0

    for weight_rel_path in tqdm(all_weights):
        net_name = weight_rel_path.split('_')[0]

        net, run_config, model_config = gen_net(net_name)

        weight_full_path = "%s/%s" % (directory.MODEL_DIR, weight_rel_path)

        model = load_model(net, weight_full_path, run_config)

        single_pred = predict(test_des, model, run_config)

        pred_list += single_pred

    res = (pred_list / len(all_weights)).tolist()

    submit(res)


if __name__ == '__main__':
    main()
