from preprocess import train_pro, test_pro
from tqdm import tqdm
import directory
import os


def gen_corpus():
    train_df = train_pro(directory.TRAIN_SET_PATH)
    test_df = test_pro(directory.TEST_SET_A_PATH)  # 用A榜测试集参与预训练

    train_num = len(train_df)
    test_num = len(test_df)

    # Text should be one-sentence-per-line, with empty lines between documents.
    with open(directory.CORPUS_PATH, 'a') as f:
        f.seek(0)
        f.truncate()

        for i in tqdm(range(train_num)):
            des_train = train_df.iloc[i, 0]

            des_train = ' '.join(str(train_item) for train_item in des_train)

            f.write(str(des_train) + '\n\n')

        for j in tqdm(range(test_num)):
            des_test = test_df.iloc[j, 0]

            des_test = ' '.join(str(test_item) for test_item in des_test)

            f.write(str(des_test) + '\n\n')

        f.close()


def gen_vocab():
    train_df = train_pro(directory.TRAIN_SET_PATH)
    test_df = test_pro(directory.TEST_SET_A_PATH)  # 用A榜测试集参与预训练

    des_train = train_df.iloc[:, 0].tolist()
    des_test = test_df.iloc[:, 0].tolist()

    des_list = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

    for sentence_train in des_train:
        for word_train in sentence_train:
            if word_train not in des_list:
                des_list.append(word_train)

    for sentence_test in des_test:
        for word_test in sentence_test:
            if word_test not in des_list:
                des_list.append(word_test)

    with open(directory.VOCAB_PATH, 'a') as f:
        f.seek(0)
        f.truncate()

        for i in tqdm(range(len(des_list))):
            f.write(str(des_list[i]) + '\n')


if __name__ == '__main__':
    if not os.path.exists(directory.DATA_DIR):
        os.makedirs(directory.DATA_DIR)

    gen_corpus()
    gen_vocab()
