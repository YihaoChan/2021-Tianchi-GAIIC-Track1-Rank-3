from pretrain.pretrain_vector import PretrainVector
from helper.spatial_dropout import SpatialDropout
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRCNNConfig(object):
    def __init__(self):
        self.hidden_size = 128

        self.num_layers = 2

        self.dropout = 0.5

        self.learning_rate = 5e-4

        self.freeze = True


class TextRCNN(nn.Module):
    def __init__(self, run_config, model_config):
        super(TextRCNN, self).__init__()

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
            input_size=embedding_size, hidden_size=model_config.hidden_size,
            num_layers=model_config.num_layers, bidirectional=True,
            batch_first=True, dropout=model_config.dropout
        )

        self.maxpool = nn.MaxPool1d(run_config.seq_len)

        self.fc = nn.Linear(model_config.hidden_size * 2 + embedding_size, 17)

        self._init_parameters()

    def forward(self, x):
        embed = self.embedding(x)

        spatial_embed = self.spatial_dropout(embed)

        out, _ = self.lstm(spatial_embed)

        out = torch.cat((embed, out), 2)

        out = F.relu(out)

        out = out.permute(0, 2, 1)

        out = self.maxpool(out).squeeze(-1)

        out = self.fc(out)

        return out

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
            else:
                nn.init.constant_(p, 0)
