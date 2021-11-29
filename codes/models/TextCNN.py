#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/23 14:54
# @Author  : lilijuan
# @File    : TextCNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from codes.utils.Model_Config import Model_Config


class Config(Model_Config):

    """配置参数"""
    def __init__(self, dataset, from_, type_, **kwargs):
        super(Config, self).__init__(dataset, from_, type_, **kwargs)
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList([nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  #x:[batch_zise,num_filters,32-filter_sizes[i]+1]
        x = F.max_pool1d(x, x.size(2)).squeeze(2) #x:[batch_zise,num_filters]
        return x

    def forward(self, x):  #x:([batch_zise,sequence_length],[batch_zise])
        out = self.embedding(x[0]) #out:([batch_zise,sequence_length,embedding_size])
        out = out.unsqueeze(1)  #out:([batch_zise,1,sequence_length,embedding_size])
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1) #out:[batch_zise,num_filters*3,sequence_length]
        out = self.dropout(out)
        out = self.fc(out)
        return out
if __name__ == '__main__':
    from codes.utils.data_loader import build_dataset, build_iterator
    from codes.train_eval import train
    config = Config(dataset="THUCNews",from_="from_pretrained_embedding",type_="char",num_epochs=20)
    model = Model(config)
    vocab, train_data, dev_data, test_data = build_dataset(config, type_='word')
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    train(config, model=model, train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter)