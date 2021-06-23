import pandas as pd
import plotly.graph_objs as go
from sklearn.manifold import TSNE
import numpy as np
import os


def plot(method, color):
    def get_3D_results():
        result_df = pd.read_csv('./submit_results/' + method + '.csv', header=None)
        result_df.columns = ['report_ID', 'probability']

        X = [list(map(float, i.strip('|').strip().split())) for i in result_df['probability'].values]

        tsne = TSNE(n_components=3, random_state=2021)
        res = tsne.fit_transform(X)

        # 归一化
        max_value = np.max(res)
        min_value = np.min(res)
        _range = max_value - min_value
        norm_res = (res - min_value) / _range

        return norm_res

    def show(data):
        x = [k for k, v, z in data]
        y = [v for k, v, z in data]
        z = [z for k, v, z in data]

        font = {'family': 'Times New Roman', 'size': 20}

        data = [
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                opacity=0.9,
                # 蓝色：#3E6BF2 深蓝：#3A2885 紫色：#8273B0 祖母绿：#009298 中蓝：#426EB4
                marker={'color': color},
            )
        ]

        layout = go.Layout(
            autosize=False,
            title=method.upper() + ' 3D Scatterplot',
            width=1000,
            height=1000,
            margin=go.layout.Margin(l=0, r=0, b=50, t=50),
            showlegend=False,
            font=font
        )

        fig = go.Figure(data=data, layout=layout)

        fig.write_image('%s/3D_results_%s.jpg' % (save_fig_path, method), width=1000, height=1000)

    three_dim_results = get_3D_results()
    show(three_dim_results)


if __name__ == '__main__':
    save_fig_path = './fig'
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)

    plot('lstm', '#426EB4')
    plot('dpcnn', '#3A2885')
    plot('han', '#8273B0')
    plot('nezha', '#009298')
    plot('merge', '#3E6BF2')
