import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def plot(set_type, color):
    df = None
    title_name = None

    if set_type == 'train':
        df = pd.read_csv("./datasets/track1_round1_train_20210222.csv", header=None)
        df.columns = ['report_ID', 'description', 'region']
        title_name = 'Train Set'
    elif set_type == 'test_a':
        df = pd.read_csv("./datasets/track1_round1_testA_20210222.csv", header=None)
        df.columns = ['report_ID', 'description']
        title_name = 'Test Set A'

    def get_2D_results():
        X = [list(map(int, i.strip('|').strip().split())) for i in df['description'].values]
        X = np.array(
            [list(r) for r in tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=100, padding='post', value=0)]
        )

        pca = PCA(n_components=2)
        res = pca.fit_transform(X)

        return res

    def show(data):
        x = [k for k, v in data]
        y = [v for k, v in data]

        plt.style.use('seaborn')

        # 蓝色：#3E6BF2 深蓝：#3A2885 紫色：#8273B0 祖母绿：#009298 中蓝：#426EB4
        plt.scatter(x, y, c=color)
        plt.xticks([])
        plt.yticks([])
        plt.title(
            'Distribution of Description for %s' % title_name,
            fontdict=dict(family='Times New Roman', weight='bold')
        )
        plt.savefig('%s/distribution_%s.jpg' % (save_fig_path, set_type), dpi=1000, transparent=True)
        plt.close()

    two_dimension_results = get_2D_results()
    show(two_dimension_results)


if __name__ == '__main__':
    save_fig_path = './fig'
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)

    plot('train', '#009298')
    plot('test_a', '#009298')
