from helper.preprocess import train_pro, test_pro
import pandas as pd
import directory
import os
from tqdm import tqdm


def nsp_region():
    train_prem_df = pd.read_csv(directory.PREM_TRAIN_SET_PATH, header=None)
    train_prem_df.columns = ['report_ID', 'description', 'region']
    train_semi_df = pd.read_csv(directory.SEMI_TRAIN_SET_PATH, header=None)
    train_semi_df.columns = ['report_ID', 'description', 'region', 'category']

    region_df = pd.concat([train_prem_df, train_semi_df], axis=0).reset_index(drop=True)
    region_df['description'] = [i.strip('|').strip() for i in region_df['description'].values]
    region_df.drop(['report_ID'], axis=1, inplace=True)

    region_groups = region_df.groupby('region')

    return region_groups


def nsp_category():
    category_df = pd.read_csv(directory.SEMI_TRAIN_SET_PATH, header=None)
    category_df.columns = ['report_ID', 'description', 'region', 'category']
    category_df = category_df.fillna(value={'category': -1})

    category_df = category_df.reset_index(drop=True)
    category_df['description'] = [i.strip('|').strip() for i in category_df['description'].values]
    category_df.drop(['report_ID'], axis=1, inplace=True)

    category_groups = category_df.groupby('category')

    return category_groups


def nsp_none():
    test_prem_a_df = pd.read_csv(directory.PREM_TEST_SET_A_PATH, header=None)
    test_prem_b_df = pd.read_csv(directory.PREM_TEST_SET_B_PATH, header=None)
    test_semi_a_df = pd.read_csv(directory.SEMI_TEST_SET_A_PATH, header=None)

    none_df = pd.concat([test_prem_a_df, test_prem_b_df, test_semi_a_df], axis=0).reset_index(drop=True)
    none_df.columns = ['report_ID', 'description']

    none_df['description'] = [i.strip('|').strip() for i in none_df['description'].values]
    none_df.drop(['report_ID'], axis=1, inplace=True)

    return none_df


def gen_corpus():
    region_groups = nsp_region()
    category_groups = nsp_category()
    none_df = nsp_none()
    
    with open(directory.CORPUS_PATH, 'w') as f:
        for _, region_group in region_groups:
            for i in range(len(region_group)):
                f.write(str(region_group.iloc[i, 0]) + '\n')

            f.write('\n')

        for _, category_group in category_groups:
            for i in range(len(category_group)):
                f.write(str(category_group.iloc[i, 0]) + '\n')

            f.write('\n')

        for i in range(len(none_df)):
            f.write(str(none_df.iloc[i, 0]) + '\n\n')


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

    with open(directory.VOCAB_PATH, 'w') as f:
        for i in tqdm(range(len(vocab))):
            f.write(str(vocab[i]) + '\n')


if __name__ == '__main__':
    if not os.path.exists(directory.DATA_DIR):
        os.makedirs(directory.DATA_DIR)

    gen_corpus()
    gen_vocab()
