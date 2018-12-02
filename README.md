# TextCNN
使用TextCNN在中文新闻数据集上进行文本分类。使用的数据集为[THUCNews](http://thuctc.thunlp.org/)的一个子集，使用的中文预训练词向量为[Chinese Word Vectors](https://github.com/Embedding/Chinese-Word-Vectors)。
## TextCNN简介
Text-CNN和传统的CNN结构类似，具有词嵌入层、卷积层、池化层和全连接层的四层结构。  

其中，Text-CNN的词嵌入层使用二维矩阵来表示长文本。词嵌入将输入文本的每个词语通过空间映射，将独热表示（One-Hot Representation）转换成分布式表示（Distributed Representation），进而可以使用低维的词向量来表示每一个词语。经过词嵌入，每个单词具有相同长度的词向量表示。将各个词语的向量表示连起来便可以得到二维矩阵。得到词向量的方式有多种，常用的是Word2vec方法。若使用预训练好的词向量，在训练模型的时候可以选择更新或不更新词向量，分别对应嵌入层状态为Non-static和Static。

Text-CNN的卷积层是主要部分，卷积核的宽度等于词向量的维度，经卷积后可以提取文本的特征向量。和在图像领域应用类似，Text-CNN可以设置多个卷积核以提取文本的多层特征，长度为N的卷积核可以提取文本中的N-gram特征。

Text-CNN的池化层一般采取Max-over-time pooling，输出最大值，从而判断词嵌入中是否含N-gram。  

Text-CNN的全连接层采用了Dropout算法以防过拟合，并使用Softmax函数输出各个类别的概率。  
## 实验过程
1. 数据集介绍
本次实验采用的数据集为THUCnews的一个子集。THUCNews是清华大学自然语言处理实验室根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。其在原始新浪新闻分类体系的基础上，重新整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。我们所采用的子集使用了其中10个分类，每个分类24000条数据，以保证数据均衡。10个分类类别如下：体育、财经、房产、家居、教育、科技、时尚、时政、 游戏、娱乐。其中训练集单个类为20000条，测试集单个类为4000条，总计训练集数据240000条，测试集40000条。  
2. 中文预训练词向量介绍
词嵌入层采用了目前最全的中文预训练词向量集合Chinese Word Vectors。Chinese Word Vectors提供使用不同表征（稀疏和密集）、上下文特征（单词、N-gram、字符等）以及语料库训练的中文词向量。  
3. 数据预处理
(1) 分词、去标点、截取短文本。
(2) 通过Keras深度学习库使用预训练的词向量训练新闻文本并将标签编码为One-Hot。  
使用预训练的词向量可以间接地引入外部的训练数据，并且能减少训练的参数，提高模型训练的效率。本实验中选择Chinese Word Vectors中以 SGNS 训练，并以单词+N-gram+字符为上下文特征，使用搜狗新闻作为语料库的密集词向量集合。该预训练词向量集含364229个词，每个词具有300维。  
由于实验中采取Softmax和交叉熵计算损失，因此需要将标签以One-Hot形式编码。  
Keras具有高度模块化、极简和可扩充特性，因此本实验采取Keras深度学习库对新闻文本进行以上的预处理。
## 实验结果
| 准确率 | 召回率 |  F1 score | 
| - | :-: | -: | 
| 97.99% | 97.98% | 97.98% | 

