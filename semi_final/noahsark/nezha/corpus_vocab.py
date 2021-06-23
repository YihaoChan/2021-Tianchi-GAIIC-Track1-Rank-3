from helper.preprocess import train_pro, test_pro
from tqdm import tqdm
import directory
import os


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
    test_semi_b_df = test_pro(directory.SEMI_TEST_SET_B_PATH)

    with open(directory.CORPUS_PATH, 'w') as f:
        write_corpus(
            f, train_prem_df, train_semi_df,
            test_prem_a_df, test_prem_b_df, test_semi_a_df, test_semi_b_df
        )

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
    test_semi_b_des = test_pro(directory.SEMI_TEST_SET_B_PATH).iloc[:, 0].tolist()

    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

    write_vocab(
        vocab, train_prem_des, train_semi_des,
        test_prem_a_des, test_prem_b_des, test_semi_a_des, test_semi_b_des
    )

    with open(directory.VOCAB_PATH, 'w') as f:
        for i in tqdm(range(len(vocab))):
            f.write(str(vocab[i]) + '\n')


if __name__ == '__main__':
    if not os.path.exists(directory.DATA_DIR):
        os.makedirs(directory.DATA_DIR)

    gen_corpus()
    gen_vocab()
