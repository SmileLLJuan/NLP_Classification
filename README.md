# NLP_Classification
nlp中基于TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention, DPCNN, Transformer等的分类任务
模型介绍：
# 1. TextCNN
## 1.1 模型结构
![Image text](https://github.com/SmileLLJuan/NLP_Classification/blob/main/images/TextCNN.png)
模型主要包括五层，第一层是embedding layer,第二层是convolutional layer,第三层是max-pooling layer,第四层是fully connected layer，最后一层是softmax layer.<br/>
其中卷积层的目的是为了特区特征，采用不同的卷积核可以提取不同的文本特征，本文采用的卷积核大小为[2,3,4]
### 卷积
卷积过程：卷积输入为[batch_size,sequence_length,embedding_dim]维度的向量矩阵，假设使用一个维度[embedding_dim,h]的卷积核W，卷积核W与Xi:i+h-1(从第i个词到第i+h-1个词)进行卷积操作在使用激活函数激活得到相应的特征ci,<br/>
则卷积操作的公式如下：
<img src="http://chart.googleapis.com/chart?cht=tx&chl= c_i=f(W \cdot X_{i:i+h-1} + b)" style="border:none;">
因此经过卷积操作之后，可以得到一个n-h+1维的向量c;<img src="http://chart.googleapis.com/chart?cht=tx&chl= C=[c_1,c_2,...,C_{n-h+1}]" style="border:none;">
<br/>
### 池化
池化的作用可以显著减少参数量，压缩数据和参数的数量，减小过拟合，同时提高模型的容错性。

## 1.2 参考文献
有几篇文章都是textcnn，模型结构类似。其中《Convolutional Neural Networks for Sentence Classification》给出了基本结构，《A Sensitivity Analysis ...》专门做了各种控制变量的实验对比。<br/>
[1][Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf) <br/>
[2][A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1510.03820.pdf)<br/>
# 2. TextRNN
## 2.1 模型结构
![Image text](https://github.com/SmileLLJuan/NLP_Classification/blob/main/images/TextRNN.png)
一般取前向/反向LSTM在最后一个时间步长上隐藏状态，然后进行拼接，在经过一个softmax层(输出层使用softmax激活函数)进行一个多分类；或者取前向/反向LSTM在每一个时间步长上的隐藏状态，对每一个时间步长上的两个隐藏状态进行拼接，然后对所有时间步长上拼接后的隐藏状态取均值，再经过一个softmax层(输出层使用softmax激活函数)进行一个多分类(2分类的话使用sigmoid激活函数)。<br/>
RNN会出现梯度消失和梯度爆炸的情况，因此难以学习到序列的长距离相关性，因此LSTM专门被提出用来解决长期依赖问题。LSTM的单元计算如下：<br/>
<img src="http://chart.googleapis.com/chart?cht=tx&chl= i_t=\delta(W_ix_t+U_ih_{t_1}+V_ic_{t-1})" style="border:none;"><br/>
<img src="http://chart.googleapis.com/chart?cht=tx&chl= f_t=\delta(W_fx_t+U_fh_{t_1}+V_fc_{t-1})" style="border:none;"><br/>
<img src="http://chart.googleapis.com/chart?cht=tx&chl= o_t=\delta(W_ox_t+U_oh_{t_1}+V_oc_{t-1})" style="border:none;"><br/>
<img src="http://chart.googleapis.com/chart?cht=tx&chl= f_t=\delta(W_fx_t+U_fh_{t_1}+V_fc_{t-1})" style="border:none;"><br/>
<img src="http://chart.googleapis.com/chart?cht=tx&chl= \tilde{c_t}=tanh(W_cx_t+U_ch_{t-1})" style="border:none;"><br/>
<img src="http://chart.googleapis.com/chart?cht=tx&chl= c_t=f_{t}^{i}\bigodot c_{t-1}+i_t\bigodot\tilde{c_t}" style="border:none;"><br/>
<img src="http://chart.googleapis.com/chart?cht=tx&chl= h_t=o_t\bigodot tanh(c_t)" style="border:none;"><br/>

## 2.2 参考文献
[1][Recurrent Neural Network for Text Classification with Multi-Task Learning](https://arxiv.org/pdf/1605.05101.pdf)<br/>



