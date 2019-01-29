#  RNN文本生成  word级别

import os
import numpy as np
import nltk
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from gensim.models.word2vec import Word2Vec
# nltk.download()
import re
from bs4 import BeautifulSoup
import pandas as pd

# 文本读入
# 第一种方式：
# raw_text = ''
# for file in os.listdir(r""):
#     if file.endswith(".txt"):
#         raw_text += open(r"" + file,errors='ignore').read() + '\n\n'

# 第二种方式：
raw_text1 = open(r"E:\algorithm_code\NLP_text_generation\data\William_McKinley.txt",encoding='utf-8').read()
raw_text2 = open(r"E:\algorithm_code\NLP_text_generation\data\Winston_Churchil.txt",encoding='utf-8').read()


def clean_text(text):
    # 读取文本 html解析 拿出具体内容
    text = BeautifulSoup(text, 'html.parser').get_text()  # 解析获取文本信息
    # 用正则表达式 把非字母的符号用空格替换掉
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower()
    return words

raw_text1 = clean_text(raw_text1)
raw_text2 = clean_text(raw_text2)

# raw_text = raw_text.lower()
sentensor = nltk.data.load('tokenizers/punkt/english.pickle') #分割句子
sents1 = sentensor.tokenize(raw_text1)
sents2 = sentensor.tokenize(raw_text2)

corpus = []  #二维数组
for sen in sents1:
    corpus.append(nltk.word_tokenize(sen)) #分词
for sen1 in sents2:
    corpus.append(nltk.word_tokenize(sen1))
print("corpus:",len(corpus))
print(corpus[:3])
# print(corpus)
print("i:",[i for i in corpus if i == 'you'])
# 使用w2v乱炖
w2v_model = Word2Vec(corpus,size=128,window=5,min_count=5,workers=4)
print(w2v_model['office']) #w2v_model里的词汇


# 接下来，其实我们还是以之前的方式来处理我们的training data，
# 把源数据变成一个长长的x，好让LSTM学会predict下一个单词：
raw_input = [item for sublist in corpus for item in sublist]
print(f"raw_input:{len(raw_input)}")
print(raw_input[12])

text_stream = []
vocab = w2v_model.wv.vocab
for word in raw_input:
    if word in vocab:
        text_stream.append(word)
print(f"text_Stream:{len(text_stream)}")


# 我们这里的文本预测就是，给了前面的单词以后，下一个单词是谁？
# 比如，hello from the other, 给出 side
#
# 构造训练测试集¶
# 我们需要把我们的raw text变成可以用来训练的x,y:
# x 是前置字母们 y 是后一个字母
seq_length = 10
x = []
y = []
for i in range(0,len(text_stream) - seq_length):
        given = text_stream[i:i+seq_length]
        predict = text_stream[i+seq_length]
        print("predict:",predict)
        print("given:",given)
        x.append(np.array([w2v_model[word] for word in given]))
        y.append(w2v_model.wv[predict])


# 看看我们做好的数据集的长相：
print("x:",x)
print("y:",y)
print("x[10]:",x[10])
print("y[10]:",y[10])

print(len(x))
print(len(y))
print(len(x[12]))
print(len(x[12][0]))
print(len(y[12]))

x = np.reshape(x,(-1,seq_length,128))
y = np.reshape(y,(-1,128))

# 接下来我们做两件事：
# 1.我们已经有了一个input的数字表达（w2v），我们要把它
# 变成LSTM需要的数组格式： [样本数，时间步伐，特征]
# 2.第二，对于output，我们直接用128维的输出
#
# 模型建造
# LSTM模型构建
from keras.models import load_model
# model = load_model(r"E:\algorithm_code\NLP_text_classification\model_word.hdf5")

model = Sequential()
model.add(LSTM(256,dropout_W=0.2,dropout_U=0.2,input_shape=(seq_length,128)))
model.add(Dropout(0.2))
model.add(Dense(128,activation='sigmoid'))
model.compile(loss='mse',optimizer='adam')

# 跑模型
model.fit(x,y,nb_epoch=100,batch_size=4096)
model.save(r"E:\algorithm_code\NLP_text_classification\model_word.hdf5")

# 看训练出来的LSTM效果
def predict_next(input_array):
    x = np.reshape(input_array,(-1,seq_length,128))
    y = model.predict(x)
    return y

def string_to_index(raw_input):
    raw_input = raw_input.lower()
    input_stream = nltk.word_tokenize(raw_input)
    res = []
    for word in input_stream[(len(input_stream) - seq_length):]:
        res.append(w2v_model[word])
    return res

def y_to_word(y):
    word = w2v_model.most_similar(positive=y,topn=1)
    return word

# 写成一个大程序
def generate_article(init,rounds=30):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_word(predict_next(string_to_index(in_string)))
        in_string += ' ' + n[0][0]
        # print(n[0][0])
    return in_string

init = 'Language Models allow us to measure how likely a sentence is which is an important for Machine'
article = generate_article(init)
print(article)