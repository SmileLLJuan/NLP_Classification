# coding: UTF-8
import torch.nn as nn
import torch.nn.functional as F
from codes.utils.Model_Config import Model_Config

class Config(Model_Config):

    """配置参数"""
    def __init__(self, dataset,from_,type_,**kwargs):
        super(Config, self).__init__(dataset,from_,type_,**kwargs)
        self.model_name = 'DPCNN'
        self.num_filters = 250                                          # 卷积核数量(channels数)


'''Deep Pyramid Convolutional Neural Networks for Text Categorization'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        #in:[batch_size,1,sequence_length,embedding_dim],out:[batch_size,num_filters,sequence_length-3+1,1]
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embed), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        x = x[0]
        x = self.embedding(x)   #[batch_size,sequence_length,embedding_dim]
        print(x.shape)
        x = x.unsqueeze(1)  # [batch_size,1,sequence_length,embedding_dim]
        print(x.shape)
        x = self.conv_region(x)  # [batch_size,num_filters,sequence_length-3+1,1]
        print(x.shape)
        x = self.padding1(x)  # [batch_size,num_filters,sequence_length,1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size,num_filters,sequence_length-3+1,1]
        x = self.padding1(x)  # [batch_size,num_filters,sequence_length,1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size,num_filters,sequence_length-3+1,1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
if __name__ == '__main__':
    from codes.utils.data_loader import build_dataset, build_iterator
    from codes.train_eval import train
    config = Config(dataset="THUCNews",from_="from_pretrained_embedding",type_="char",num_epochs=10,device='cpu',batch_size=128)
    vocab, train_data, dev_data, test_data = build_dataset(config, ues_word=False)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    model = Model(config)
    train(config, model=model, train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter)
