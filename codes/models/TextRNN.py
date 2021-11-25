# coding: UTF-8
import torch.nn as nn

from codes.utils.Model_Config import Model_Config
class Config(Model_Config):

    """配置参数"""
    def __init__(self, dataset, embedding,**kwargs):
        super(Config, self).__init__(dataset,embedding,**kwargs)
        self.model_name = 'TextRNN'
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数


'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        # self.lstm = nn.RNN(config.embed, config.hidden_size, config.num_layers,bidirectional=True, batch_first=True, dropout=config.dropout)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,bidirectional=True, batch_first=True, dropout=config.dropout)
        # self.lstm = nn.GRU(config.embed, config.hidden_size, config.num_layers,bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)  # [batch_size, seq_len, embedding_dim]=[128, 32, 300]
        out, _ = self.lstm(out) #out:[batch_size, seq_len,hidden_size*2],_:([4,batch_size,hidden_size],[4,batch_size,hidden_size])
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

    '''变长RNN，效果差不多，甚至还低了点...'''
    # def forward(self, x):
    #     x, seq_len = x
    #     out = self.embedding(x)
    #     _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从长到短排序（index）
    #     _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index
    #     out = torch.index_select(out, 0, idx_sort)
    #     seq_len = list(seq_len[idx_sort])
    #     out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
    #     # [batche_size, seq_len, num_directions * hidden_size]
    #     out, (hn, _) = self.lstm(out)
    #     out = torch.cat((hn[2], hn[3]), -1)
    #     # out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    #     out = out.index_select(0, idx_unsort)
    #     out = self.fc(out)
    #     return out
if __name__ == '__main__':
    from codes.utils.data_loader import build_dataset, build_iterator
    from codes.train_eval import train

    config = Config(dataset="THUCNews", embedding="embedding_SougouNews.npz", num_epochs=10)
    vocab, train_data, dev_data, test_data = build_dataset(config, ues_word=False)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    model = Model(config)
    train(config, model=model, train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter)
