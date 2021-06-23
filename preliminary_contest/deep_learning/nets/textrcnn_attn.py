from pretrain.pretrain_vector import PretrainVector
from helper.spatial_dropout import SpatialDropout
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRCNNAttnConfig(object):
    def __init__(self):
        self.hidden_size1 = 128

        self.hidden_size2 = 64

        self.num_layers = 2

        self.dropout = 0.5

        self.learning_rate = 5e-4

        # 训练过程中是否冻结对词向量的更新
        self.freeze = True


class TextRCNNAttn(nn.Module):
    def __init__(self, run_config, model_config):
        super(TextRCNNAttn, self).__init__()

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

        self.lstm = nn.GRU(
            input_size=embedding_size, hidden_size=model_config.hidden_size1,
            num_layers=model_config.num_layers, bidirectional=True,
            batch_first=True, dropout=model_config.dropout
        )

        self.tanh = nn.Tanh()

        self.w = nn.Parameter(torch.zeros(model_config.hidden_size1 * 2))

        self.fc1 = nn.Linear(model_config.hidden_size1 * 2 + embedding_size, model_config.hidden_size2)

        self.maxpool = nn.MaxPool1d(run_config.seq_len)

        self.fc2 = nn.Linear(model_config.hidden_size2, 17)

        self._init_parameters()

    def forward(self, x):
        embed = self.embedding(x)

        spatial_embed = self.spatial_dropout(embed)

        H, _ = self.lstm(spatial_embed)

        M = self.tanh(H)

        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)

        out = H * alpha

        out = torch.cat((embed, out), 2)

        out = F.relu(out)

        out = out.permute(0, 2, 1)

        out = self.maxpool(out).squeeze(-1)

        out = self.fc1(out)

        out = self.fc2(out)

        return out

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
            else:
                nn.init.constant_(p, 0)
