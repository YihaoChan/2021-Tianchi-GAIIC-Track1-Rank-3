# 可视化

## 1 文件结构

```
├── datasets-------------------------初赛数据集
│   ├── track1_round1_testA_20210222.csv
│   ├── track1_round1_testB.csv
│   └── track1_round1_train_20210222.csv
├── eda.py----------------------------------------------分析句长、词典、词频、标签分布
├── plot_2D_distribution.py-----------------------------PCA二维可视化数据分布
├── plot_3D_results.py----------------------------------t-SNE三维可视化推理结果
└── submit_results-------------------初赛提交结果
    ├── dpcnn.csv
    ├── han.csv
    ├── lstm.csv
    ├── merge.csv
    └── nezha.csv
```

## 2 执行环境

```
conda创建python3.6虚拟环境
conda install -c plotly plotly-orca 
(-c 即 -channel，频道是Navigator和conda查找包的位置，具有相同名称的包可能存在于多个通道上，如果希望从默认通道以外的其他通道安装，则指定要使用哪个通道的一种方法是使用 conda install -c channel_name package_name语法。)
plotly==4.14.3
psutil==5.8.0
requests==2.25.1
retrying==1.3.3
tensorflow==1.15.0
tensorflow-gpu==1.15.0
scikit-learn==0.24.2
pandas==0.20.3
matplotlib==3.3.4
numpy==1.19.4
```
