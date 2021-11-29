# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.utils.Model_Config import Model_Config
class Config(Model_Config):

    """配置参数"""
    def __init__(self, dataset, from_, type_, **kwargs):
        super(Config, self).__init__(dataset, from_, type_, **kwargs)
        self.model_name = 'TextRNN_Att'
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.hidden_size2 = 64


'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,bidirectional=True, batch_first=True, dropout=config.dropout).to(device=config.device)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2)).to(device=config.device)
        self.tanh2 = nn.Tanh().to(device=config.device)
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2).to(device=config.device)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes).to(device=config.device)

    def forward(self, x):
        x, _ = x
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out
if __name__ == '__main__':
    from codes.utils.data_loader import build_dataset, build_iterator
    from codes.train_eval import train

    config = Config(dataset="THUCNews", from_="from_pretrained_embedding", type_="char", num_epochs=10,device='cuda')
    print(config.embedding_pretrained.shape,config.embedding_pretrained.device)
    vocab, train_data, dev_data, test_data = build_dataset(config, ues_word=False)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    model = Model(config)
    train(config, model=model, train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter)
