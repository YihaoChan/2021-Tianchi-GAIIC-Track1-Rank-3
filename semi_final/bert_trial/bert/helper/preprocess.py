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
    train_df['region'] = [i.strip('|').strip() for i in train_df['region'].values]

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
