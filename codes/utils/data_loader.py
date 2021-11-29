#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/23 15:05
# @Author  : lilijuan
# @File    : data_loader.py
import os
import torch
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import random
from codes.utils.vocab_embedding import get_tokenizer,PAD,UNK,Vocab



def build_dataset(config, type_='word',pad_size=32):
    tokenizer=get_tokenizer(type_=type_, language="zh")
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    # else:
    #     vocab = Vocab(from_='from_train_data', type_=type_, train_dir=config.train_path)
    #     word_to_id, embeddings=vocab.build_from_train_data(train_dir=config.train_path)
    #     pkl.dump(word_to_id, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")
    def load_dataset(path, pad_size=pad_size):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f.readlines()):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        random.shuffle(contents) #list乱序
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    from codes.models.TextCNN import Config

    config = Config(dataset="THUCNews", from_="from_pretrained_embedding", type_="char",batch_size=2)
    vocab, train_data, dev_data, test_data=build_dataset(config,type_='word',pad_size=16)
    print(len(vocab),train_data[:2])
    train_iter = build_iterator(train_data, config)
    print(train_iter.__next__())
