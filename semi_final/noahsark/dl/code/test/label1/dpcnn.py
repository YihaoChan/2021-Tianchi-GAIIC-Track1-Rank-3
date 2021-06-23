import torch
import torch.nn as nn
from config import DPCNNConfig
import torch.nn.functional as F


class DPCNN(nn.Module):

    def __init__(self, embed_num, static=False):
        super(DPCNN, self).__init__()

        embed_dim = DPCNNConfig.embed_dim
        class_num = 17
        num_filters = DPCNNConfig.num_filters
        dropout = DPCNNConfig.dropout

        self.embed = nn.Embedding(embed_num, embed_dim)  # 词嵌入
        self.conv_region = nn.Conv2d(1, num_filters, (3, embed_dim), stride=1)
        self.conv = nn.Conv2d(num_filters, num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, class_num)

    def forward(self, x):
        # x = x[0]
        x = self.embed(x)
        # print("embed(x).shape:",x.shape)
        x = x.unsqueeze(1)  # [batch_size, num_filters, seq_len, 1]
        # print("x.unsqueeze:",x.shape)
        x = self.conv_region(x)  # [batch_size, num_filters, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters]
        x = self.dropout(x)
        x = self.fc(x)
        # print(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
