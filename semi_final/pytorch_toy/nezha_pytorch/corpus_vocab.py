import os
import pandas as pd
import directory
from tqdm import tqdm


def train_pro(train_set_path):
    if train_set_path not in [directory.PREM_TRAIN_SET_PATH, directory.SEMI_TRAIN_SET_PATH]:
        raise ValueError('Train_set_path is wrong!')

    train_df = pd.read_csv(train_set_path, header=None)

    if train_set_path == directory.SEMI_TRAIN_SET_PATH:
        train_df.columns = ['report_ID', 'description', 'region', 'category']
    elif train_set_path == directory.PREM_TRAIN_SET_PATH:
        train_df.columns = ['report_ID', 'description', 'region']

    train_df.drop(['report_ID'], axis=1, inplace=True)
    train_df['description'] = [i.strip('|').strip() for i in train_df['description'].values]

    train_num = len(train_df)

    for train_idx in tqdm(range(train_num)):
        des = train_df.loc[train_idx, 'description']

        des = [int(word) for word in des.split(' ')]

        train_df.loc[train_idx, 'description'] = des

    return train_df


def test_pro(test_set_path):
    if test_set_path not in [
        directory.PREM_TEST_SET_A_PATH, directory.PREM_TEST_SET_B_PATH, directory.SEMI_TEST_SET_A_PATH
    ]:
        raise ValueError('Test_set_path is wrong!')

    test_df = pd.read_csv(test_set_path, header=None)

    test_df.columns = ['report_ID', 'description']

    test_df.drop(['report_ID'], axis=1, inplace=True)

    test_df['description'] = [i.strip('|').strip() for i in test_df['description'].values]

    test_num = len(test_df)

    for test_idx in tqdm(range(test_num)):
        des = test_df.loc[test_idx, 'description']

        des = [int(word) for word in des.split(' ')]

        test_df.loc[test_idx, 'description'] = des

    return test_df


def gen_corpus():
    def write_corpus(corpus_file, *dfs):
        print('Total number of corpus file components: ' + str(len(dfs)))

        for df in dfs:
            for idx in range(len(df)):
                des = df.iloc[idx, 0]
                des = ' '.join(str(item) for item in des)
                # Text should be one-sentence-per-line, with empty lines between documents.
                corpus_file.write(str(des) + '\n\n')

    train_prem_df = train_pro(directory.PREM_TRAIN_SET_PATH)
    train_semi_df = train_pro(directory.SEMI_TRAIN_SET_PATH)
    test_prem_a_df = test_pro(directory.PREM_TEST_SET_A_PATH)
    test_prem_b_df = test_pro(directory.PREM_TEST_SET_B_PATH)
    test_semi_a_df = test_pro(directory.SEMI_TEST_SET_A_PATH)

    with open(directory.CORPUS_PATH, 'a') as f:
        f.seek(0)
        f.truncate()

        write_corpus(f, train_prem_df, train_semi_df, test_prem_a_df, test_prem_b_df, test_semi_a_df)

        f.close()


def gen_vocab():
    def write_vocab(vocab_list, *deses):
        print('Total number of vocab file components: ' + str(len(deses)))

        for des in deses:
            for sentence in des:
                for word_train in sentence:
                    if word_train not in vocab_list:
                        vocab_list.append(word_train)

    train_prem_des = train_pro(directory.PREM_TRAIN_SET_PATH).iloc[:, 0].tolist()
    train_semi_des = train_pro(directory.SEMI_TRAIN_SET_PATH).iloc[:, 0].tolist()
    test_prem_a_des = test_pro(directory.PREM_TEST_SET_A_PATH).iloc[:, 0].tolist()
    test_prem_b_des = test_pro(directory.PREM_TEST_SET_B_PATH).iloc[:, 0].tolist()
    test_semi_a_des = test_pro(directory.SEMI_TEST_SET_A_PATH).iloc[:, 0].tolist()

    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

    write_vocab(vocab, train_prem_des, train_semi_des, test_prem_a_des, test_prem_b_des, test_semi_a_des)

    with open(directory.VOCAB_PATH, 'a') as f:
        f.seek(0)
        f.truncate()

        for i in tqdm(range(len(vocab))):
            f.write(str(vocab[i]) + '\n')


if __name__ == '__main__':
    if not os.path.exists(directory.DATA_DIR):
        os.makedirs(directory.DATA_DIR)

    if not os.path.exists(directory.PRETRAIN_DIR):
        os.makedirs(directory.PRETRAIN_DIR)

    gen_corpus()
    gen_vocab()
