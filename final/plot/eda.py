import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

font = {
    'family': 'Times New Roman',
    'weight': 'bold',
    'size': 20
}

plt.style.use('seaborn')


def plot_train_seq_len():
    train_df['train_description_len'] = train_df['description'].apply(lambda x: len(x.split(' ')))
    plt.figure(figsize=(10, 8), dpi=800)
    _ = plt.hist(train_df['train_description_len'], bins=300)
    plt.tick_params(axis='both', labelsize=20)
    plt.xlabel('Sequence Length', fontdict=font)
    plt.ylabel('Number of Descriptions', fontdict=font)
    plt.title("Distribution of Sequence Length for Train Set", fontdict=font)
    plt.savefig('%s/seq_len_train_set.jpg' % save_fig_path, dpi=800, transparent=True)
    plt.close()


def plot_test_a_seq_len():
    test_df_A['test_description_len'] = test_df_A['description'].apply(lambda x: len(x.split(' ')))
    plt.figure(figsize=(10, 8), dpi=800)
    _ = plt.hist(test_df_A['test_description_len'], bins=300)
    plt.tick_params(axis='both', labelsize=20)
    plt.xlabel('Sequence Length', fontdict=font)
    plt.ylabel('Number of Descriptions', fontdict=font)
    plt.title("Distribution of Sequence Length for Test Set A", fontdict=font)
    plt.savefig('%s/seq_len_test_set_a.jpg' % save_fig_path, dpi=800, transparent=True)
    plt.close()


def count_word_freq_label_distrbution():
    # vocab_size
    with open(train_file, 'r') as f:
        lines = f.readlines()
    train_texts, train_labels = [], []
    for id, line in enumerate(lines):
        line = line.strip().replace('|', '').split(',')
        text = line[1].strip().split(' ')
        text = [int(word) for word in text]
        train_texts.append(text)
        train_labels.append(line[2])

    with open(test_a_file, 'r') as f:
        lines = f.readlines()
    test_texts = []
    for id, line in enumerate(lines):
        line = line.strip().replace('|', '').split(',')
        text = line[1].strip().split(' ')
        text = [int(word) for word in text]
        test_texts.append(text)

    train_vocab_size = max(max(text) for text in train_texts) - min(min(text) for text in train_texts) + 1
    test_vocab_size = max(max(text) for text in test_texts) - min(min(text) for text in test_texts) + 1
    print("训练集的vocab_size: {}".format(train_vocab_size))
    print("测试集的vocab_size: {}".format(test_vocab_size))

    # 统计词频
    print('-----训练集词频-----')
    word_count = np.zeros(train_vocab_size, dtype='int32')
    for text in train_texts:
        for word in text:
            word_count[word] += 1
    sorted_index = word_count.argsort()

    for i in range(-1, -21, -1):
        word = sorted_index[i]
        print(str(word) + "|" + str(word_count[word]))

    print('-----测试集词频-----')
    word_count = np.zeros(test_vocab_size, dtype='int32')
    for text in test_texts:
        for word in text:
            word_count[word] += 1
    sorted_index = word_count.argsort()

    for i in range(-1, -21, -1):
        word = sorted_index[i]
        print(str(word) + "|" + str(word_count[word]))

    # 标签分布
    print('-----训练集标签分布-----')
    label_counts = np.zeros(17, dtype='int32')
    for labels in train_labels:
        labels = labels.strip().split(' ')
        for label in labels:
            if label != '':
                label_counts[int(label)] += 1
    for label, counts in enumerate(label_counts):
        print('%d|%d' % (label, counts))


if __name__ == '__main__':
    train_file = './datasets/track1_round1_train_20210222.csv'
    test_a_file = './datasets/track1_round1_testA_20210222.csv'

    train_df = pd.read_csv('./datasets/track1_round1_train_20210222.csv', header=None)
    test_df_A = pd.read_csv('./datasets/track1_round1_testA_20210222.csv', header=None)

    train_df.columns = ['report_ID', 'description', 'label']
    test_df_A.columns = ['report_ID', 'description']
    train_df.drop(['report_ID'], axis=1, inplace=True)
    test_df_A.drop(['report_ID'], axis=1, inplace=True)

    train_df['description'] = [i.strip('|').strip() for i in train_df['description'].values]
    train_df['label'] = [i.strip('|').strip() for i in train_df['label'].values]
    test_df_A['description'] = [i.strip('|').strip() for i in test_df_A['description'].values]

    save_fig_path = './fig'
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)

    plot_train_seq_len()
    plot_test_a_seq_len()
    count_word_freq_label_distrbution()
