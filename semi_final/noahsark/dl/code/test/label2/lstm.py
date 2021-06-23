import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LSTMConfig
from torch.autograd import Variable


class BilstmAttention(nn.Module):

    def __init__(self, embed_num, static=False):  # embed_num=859
        super(BilstmAttention, self).__init__()
        embed_dim = LSTMConfig.embed_dim
        class_num = 12
        word_hidden_size = LSTMConfig.word_hidden_size
        # word_num_layers=2

        self.embed = nn.Embedding(embed_num, embed_dim)  # 词嵌入
        # print("_init_self.embed",self.embed)
        self.lstm = nn.LSTM(embed_dim, word_hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(0.15)
        self.fc1 = nn.Linear(word_hidden_size * 2, class_num)

        # if static:
        #   self.embed.weight.requires_grad = False

    def attention_net(self, lstm_output, final_state):
        word_hidden_size = LSTMConfig.word_hidden_size
        hidden = final_state.view(-1, word_hidden_size * 2,
                                  1)  # hidden : [batch_size,word_hidden_size * num_directions(=2), 1(=n_layer)]
        # print("hidden.shape:",hidden.shape)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        # print("attn_weights.shape:",attn_weights.shape)
        soft_attn_weights = F.softmax(attn_weights, 1)
        # print("soft_attn_weights.shape:",soft_attn_weights.shape)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, word_hidden_size * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        # print("context.shape:",context.shape)
        return context, soft_attn_weights.data.cpu().numpy()  # context : [batch_size, word_hidden_size * num_directions(=2)]

    def forward(self, x):
        word_hidden_size = LSTMConfig.word_hidden_size
        input = self.embed(x)  # [batch_size, len_seq, embedding_dim]
        # print("self.embed(x).shape:",x.shape)
        input = input.transpose(0, 1)  # [ len_seq, batch_size, word_hidden_size]
        hidden_state = Variable(torch.zeros(1 * 2, len(x),
                                            word_hidden_size))  # [num_layers(=1) * num_directions(=2), batch_size, word_hidden_size]
        # print("hidden_state.shape:",hidden_state.shape)
        hidden_state = hidden_state.cuda()
        cell_state = Variable(torch.zeros(1 * 2, len(x),
                                          word_hidden_size))  # [num_layers(=1) * num_directions(=2), batch_size, word_hidden_size]
        # print("cell_state.shape:",cell_state.shape)
        cell_state = cell_state.cuda()
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        # print("output.shape:",output.shape)
        # print("final_hidden_state.shape:",final_hidden_state.shape)
        # print("final_cell_state.shape:",final_cell_state.shape)
        output = output.transpose(0, 1)  # [batch_size, len_seq, word_hidden_size*2]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        # print("attn_output:",attn_output
        # print("attention:",attention)
        # attn_output,_ = self.attention_net(output, final_hidden_state)
        attn_output = self.dropout(attn_output)  #
        logit = self.fc1(attn_output)  # [ batch_size, class_num]
        # print(logit.shape)
        return logit
