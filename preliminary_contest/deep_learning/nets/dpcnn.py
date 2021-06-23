from pretrain.pretrain_vector import PretrainVector
from helper.spatial_dropout import SpatialDropout
import torch.nn as nn
import torch
import torch.nn.functional as F


class DPCNNConfig(object):
    def __init__(self):
        self.dropout = 0.5

        self.learning_rate = 5e-4

        self.num_filters = 200

        # 训练过程中是否冻结对词向量的更新
        self.freeze = True


class DPCNN(nn.Module):
    def __init__(self, run_config, model_config):
        super(DPCNN, self).__init__()

        # 加载预训练的词向量
        weight_vector = PretrainVector().load_pretrained_vec(vec_type='concat')

        # 词向量维度作为embedding层维度
        embedding_size = weight_vector.shape[1]

        # 填充的字符用常数向量表示
        pad_char_vector = torch.full([1, embedding_size], fill_value=858, dtype=torch.long)

        # 拼接成嵌入向量
        embedding_vector = torch.cat((weight_vector, pad_char_vector), dim=0)

        # 把填充词当做零向量
        self.embedding = nn.Embedding.from_pretrained(embedding_vector, padding_idx=858)

        self.embedding.weight.requires_grad = False if model_config.freeze else True

        self.spatial_dropout = SpatialDropout(drop_prob=0.5)

        self.conv_region = nn.Conv2d(1, model_config.num_filters, (3, embedding_size))

        self.conv = nn.Conv2d(model_config.num_filters, model_config.num_filters, (3, 1))

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)

        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))

        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))

        self.relu = nn.ReLU()

        self.fc = nn.Linear(model_config.num_filters, 17)

        self._init_parameters()

    def forward(self, x):
        x = self.embedding(x)

        x = self.spatial_dropout(x)

        x = x.unsqueeze(1)

        x = self.conv_region(x)

        x = self.padding1(x)

        x = self.relu(x)

        x = self.conv(x)

        x = self.padding1(x)

        x = self.relu(x)

        x = self.conv(x)

        while x.size()[2] > 2:
            x = self._block(x)

        x = x.squeeze(-1)
        x = x.squeeze(-1)

        x = self.fc(x)

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

        x = x + px

        return x

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
            else:
                nn.init.constant_(p, 0)
