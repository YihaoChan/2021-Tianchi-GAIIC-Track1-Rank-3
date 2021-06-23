class LSTMConfig(object):
    # 训练参数
    max_epoch = 15
    n_splits = 10
    train_batch_size = 16
    val_batch_size = 16
    lr = 2e-3
    # CosineAnnealingWarmRestarts中的参数
    T_0 = 3
    T_mult = 2
    # datasets参数
    seq_len = 55
    # 模型参数
    embed_dim = 128
    word_hidden_size = 256


class HANConfig(object):
    # 训练参数
    max_epoch = 15
    n_splits = 10
    train_batch_size = 16
    val_batch_size = 16
    # datasets参数
    seq_len = 56
    sen_number = 28  # 每个描述分成sen_number个句子
    words_per_sen = seq_len / sen_number  # 每个句子包含的字数，seq_len必须为sen_number的整数倍
    # 模型参数
    emb_size = 128  # 词嵌入维数
    word_rnn_size = 256  # size of (bidirectional) word-level RNN
    sentence_rnn_size = 128  # size of (bidirectional) sentence-level RNN
    word_rnn_layers = 2  # number of layers in word-level RNN
    sentence_rnn_layers = 2  # number of layers in sentence-level RNN
    word_att_size = 128  # size of word-level attention layer
    sentence_att_size = 128  # size of sentence-level attention layer
    dropout = 0.5


class DPCNNConfig(object):
    # 训练参数
    max_epoch = 15
    n_splits = 10
    train_batch_size = 16
    val_batch_size = 16
    lr = 1e-3
    # datasets参数
    seq_len = 55
    # 模型参数
    embed_dim = 128
    num_filters = 256
    dropout = 0.4
