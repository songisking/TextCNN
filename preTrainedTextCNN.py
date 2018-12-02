from __future__ import print_function
import jieba
import jieba.posseg as pseg
import pandas as pd
import tensorflow as tf
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Dropout, LSTM
from keras.models import Model
from keras.initializers import Constant

MAX_NEWS_LENGTH = 200   #每个文本的最大长度
VECTOR_DIR = '~/huangjiawei/textClassification/sgns.sogounews.bigram-char'
TRAIN_DATA_URL = '~/huangjiawei/textClassification/NewData/train.csv'
TEST_DATA_URL = '~/huangjiawei/textClassification/NewData/test.csv'
MODEL_URL = '~/huangjiawei/textClassification/text-cnn/text-cnn.h5'
MAX_NUM_WORDS = 300000     #用于构建词向量的词汇表数量


# 读入数据
def readData(filePath):
    '''
    :param filePath: path of raw data
    :return: dataframe
    '''
    df = pd.read_csv(filePath)
    return df


# 单条新闻预处理：分词 去标点 取前100个词
def cut(str):
    '''
    :param str: news content(str type)
    :return: words(list type)
    '''
    data = pseg.lcut(str)
    data_list = []
    for item in data:
        if item.flag == 'x':
            data.remove(item)
        else:
            data_list.append(item.word)
    return data_list[0:MAX_NEWS_LENGTH]


# 对df的content列所有新闻进行处理
def preprocess(url):
    '''
    :param url: data url
    :return: dataframe
    '''
    df = readData(url)
    content = df['content']
    c_list = []
    for data in content:
        content_list = cut(data)
        c_list.append(content_list)
    content_short = pd.Series(c_list)
    df['content_short'] = content_short
    return df


# 将预处理后的train和test保存
def process():
    preurl = r'D:\competition\Cnews\cnews_train.csv'
    df = preprocess(preurl)
    posturl = r'D:\competition\Cnews\cnews_train_cut.csv'
    df.to_csv(posturl)
    preurl2 = r'D:\competition\Cnews\cnews_test.csv'
    df = preprocess(preurl2)
    posturl2 = r'D:\competition\Cnews\cnews_test_cut.csv'
    df.to_csv(posturl2)


# 将词向量文件保存至字典，'word_name':embedding
## 读入词向量文件，文件中的每一行的第一个变量是单词，后面的一串数字对应这个词的词向量
chineseWordVector_dir = VECTOR_DIR
f = open(chineseWordVector_dir, "r", encoding="utf-8")
## 获取词向量的维度,l表示单词数，w为某个单词转化为词向量后的维度,注意，部分预训练好的词向量的第一行并不是该词向量的维度
l, embeddings_dim = f.readline().split()
## 创建词向量索引字典
embeddings_index = {}
lines = f.readlines()
for index,line in enumerate(lines):
    ## 读取词向量文件中的每一行
    values = line.split()
    word = values[0]
    try:
        coefs = np.asarray(values[1:], dtype="float32")
        ## 将读入的这行词向量加入词向量索引字典
        embeddings_index[word] = coefs
    except:
        print(index)
        word = values[0]+values[1]
        coefs = np.asarray(values[2:], dtype="float32")
        ## 将读入的这行词向量加入词向量索引字典
        embeddings_index[word] = coefs

f.close()
print('Found %s word vectors.' % len(embeddings_index))  # 365112


# 载入数据
train_data = preprocess(TRAIN_DATA_URL)
test_data = preprocess(TEST_DATA_URL)
data = train_data.append(test_data)
## 将标签映射为数字
labels_index = {}
labels = []
texts = list(np.array(data['content_short']))
for name in set(data['classification']):
    label_id = len(labels_index)
    labels_index[name] = label_id
for data in data['classification']:
    label = labels_index[data]
    labels.append(label)

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)  #将多个文档转换为word下标的向量形式,shape为[len(texts)，len(text)]
word_index = tokenizer.word_index  # 一个dict，保存所有word对应的编号id，从1开始

print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_NEWS_LENGTH)
labels = to_categorical(np.asarray(labels))  #将标签处理为one-hot向量

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

num_validation_samples = int(train_data.shape[0])
x_train = data[:num_validation_samples]
y_train = labels[:num_validation_samples]
x_val = data[num_validation_samples:]
y_val = labels[num_validation_samples:]

# 构建词向量矩阵，预训练的词向量中没有出现的词用0向量表示
## 创建一个0矩阵，这个向量矩阵的大小为（词汇表的长度+1，词向量维度）
print('Preparing embedding matrix.')
num_words = min(MAX_NUM_WORDS, len(word_index)+ 1)
print(num_words)
embedding_matrix = np.zeros((int(num_words), int(embeddings_dim)))
## 遍历词汇表中的每一项
count = 0
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    ## 在词向量索引字典中查询单词word的词向量
    embedding_vector = embeddings_index.get(word)
    ## 判断查询结果，如果查询结果不为空,用该向量替换0向量矩阵中下标为i的那个向量
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        count += 1
print(count)


# 搭建模型
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(int(num_words), int(embeddings_dim),
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_NEWS_LENGTH,
                            trainable=True

                            )

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_NEWS_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Dropout(0.4)(embedded_sequences)
x = Conv1D(25, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Flatten()(x)
x = Dense(25, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=5,
          validation_data=(x_val, y_val
                        ))
model.summary()
model.save(MODEL_URL)

