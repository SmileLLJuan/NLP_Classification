<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# NLP_Classification
nlp中基于TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention, DPCNN, Transformer等的分类任务
模型介绍：
# 1. TextCNN
## 1.1 模型结构
![Image text](https://github.com/SmileLLJuan/NLP_Classification/blob/main/images/TextCNN.png)
模型主要包括五层，第一层是embedding layer,第二层是convolutional layer,第三层是max-pooling layer,第四层是fully connected layer，最后一层是softmax layer.<br/>
其中卷积层的目的是为了特区特征，采用不同的卷积核可以提取不同的文本特征，本文采用的卷积核大小为[2,3,4]
卷积过程：卷积输入为[batch_size,sequence_length,embedding_dim]维度的向量矩阵，假设使用一个维度[embedding_dim,h]的卷积核W，卷积核W与Xi:i+h-1(从第i个词到第i+h-1个词)进行卷积操作在使用激活函数激活得到相应的特征ci,则卷积操作的公式如下：
<u><img src="http://chart.googleapis.com/chart?cht=tx&chl=$c_i=f(W\cdot X_{i:i+h-1}+b)$" style="border:none;"></u>

池化的作用可以显著减少参数量，压缩数据和参数的数量，减小过拟合，同时提高模型的容错性。

## 1.2 参考文献
有几篇文章都是textcnn，模型结构类似。其中《Convolutional Neural Networks for Sentence Classification》给出了基本结构，《A Sensitivity Analysis ...》专门做了各种控制变量的实验对比。<br/>
[1][Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf) <br/>
[2][A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1510.03820.pdf)<br/>

