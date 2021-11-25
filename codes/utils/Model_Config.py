#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 9:58
# @Author  : lilijuan
# @File    : Model_Config.py
import os
import torch
import numpy as np
class Model_Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding="random",**kwargs):
        self.model_name = 'TextCNN'
        dataset=(kwargs['data_path'] if 'data_path' in kwargs else "../../data/") +dataset
        print(os.path.abspath(dataset + '/data/class.txt'))
        self.train_path = dataset + '/data/train.txt' if 'train_path' not in kwargs else kwargs['train_path']  # 训练集
        self.dev_path = dataset + '/data/dev.txt'if 'dev_path' not  in kwargs else kwargs['dev_path'] # 验证集
        self.test_path = dataset + '/data/test.txt' if 'test_path' not in kwargs else kwargs['test_path'] # 测试集
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/embeddings/vocab.pkl'if 'vocab_path' not  in kwargs else kwargs['vocab_path'] # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt' if 'save_path' not  in kwargs else kwargs['save_path'] # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name if 'log_path' not  in kwargs else kwargs['log_path']
        self.device = torch.device('cuda' if torch.cuda.is_available() and "device" in kwargs and kwargs['device']=='cuda'else 'cpu')   # 设备

        if "embedding"!="random" and os.path.exists(dataset + '/embeddings/' + embedding):# 预训练词向量
            self.embedding_pretrained = torch.tensor(np.load(dataset + '/embeddings/' + embedding)["embeddings"].astype('float32')).to(device=self.device)
        else:
            self.embedding_pretrained=None

        self.dropout = 0.5 if 'dropout'not in kwargs else kwargs['dropout'] # 随机失活
        self.require_improvement = 1000 if 'require_improvement'not in kwargs else kwargs['require_improvement']# 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20 if 'num_epochs'not in kwargs else kwargs['num_epochs']# epoch数
        self.batch_size = 128 if 'batch_size'not in kwargs else kwargs['batch_size']# mini-batch大小
        self.pad_size = 32 if 'pad_size'not in kwargs else kwargs['pad_size']# 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3 if 'learning_rate'not in kwargs else kwargs['learning_rate']                                      # 学习率
        self.embed = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300           # 字向量维度
