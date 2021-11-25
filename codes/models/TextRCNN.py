# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.utils.Model_Config import Model_Config
class Config(Model_Config):

    """配置参数"""
    def __init__(self, dataset, embedding,**kwargs):
        super(Config, self).__init__(dataset,embedding,**kwargs)
        self.model_name = 'TextRCNN'
        self.hidden_size = 256                                          # lstm隐藏层
        self.num_layers = 1                                             # lstm层数


'''Recurrent Convolutional Neural Networks for Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)

    def forward(self, x):
        x, _ = x
        embed = self.embedding(x)  # [batch_size, seq_len, embedding_dim]=[64, 32, 64]
        out, _ = self.lstm(embed) # [batch_size, seq_len, hidden_size*2]
        out = torch.cat((embed, out), 2) # [batch_size, seq_len, embedding_dim+hidden_size*2]
        out = F.relu(out)
        out = out.permute(0, 2, 1) # [batch_size, embedding_dim+hidden_size*2,seq_len]
        out = self.maxpool(out).squeeze() # [batch_size, embedding_dim+hidden_size*2,1]->[batch_size, embedding_dim+hidden_size*2]
        out = self.fc(out)
        return out
if __name__ == '__main__':
    from codes.utils.data_loader import build_dataset, build_iterator
    from codes.train_eval import train
    config = Config(dataset="THUCNews", embedding="embedding_SougouNews.npz", num_epochs=10,device='cpu')
    vocab, train_data, dev_data, test_data = build_dataset(config, ues_word=False)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    model = Model(config)
    train(config, model=model, train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter)