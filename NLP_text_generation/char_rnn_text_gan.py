#  用RNN做文本生成  char(字符级别)

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# 文本读入
raw_text = open(r"E:\algorithm_code\rnn_text_generated\data\Winston_Churchil.txt").read()
raw_text = raw_text.lower()

# 既然我们是以每个字母为层级，字母总共才26个，
# 所以我们可以很方便的用One-Hot来编码出所有的字母（当然，可能还有些标点符号和其他noise）
chars = sorted(list(set(raw_text)))
char_to_int = dict((c,i) for i,c in enumerate(chars))
int_to_char = dict((i,c) for i,c in enumerate(chars))

# 查看全部chars
print(chars)
print(f"一共有：{len(chars)}")
print(f"原本本一共有：{len(raw_text)}")

# 这里简单的文本预测就是，给了前置的字母以后，下一个字母是谁？
#  比如，Winsto, 给出 n Britai 给出 n


# 构建训练测试集
# 我们需要把我们的raw_text变成可以用来训练的x，y
# x是前置字母，y是后一个字母
seq_length = 100
x = []
y = []
for i in range(0,len(raw_text) - seq_length):
    given = raw_text[i:i+seq_length]
    predict = raw_text[i+seq_length]
    x.append([char_to_int[char] for char in given])
    y.append(char_to_int[predict])

# 查看数据集长相
print(f"x:{x[:3]} '\n' y:{y[:3]}")
# 查看到的数据表达方式，类似就是一个词袋，或者说index
# 接下来我们做两件事：
# 1.我们已经有了一个input的数字表达（index），我们要把它变成LSTM需要的数组格式： [样本数，时间步伐，特征]
# 2.第二，对于output，我们在Word2Vec里学过，用one-hot做output的预测可以给我们更好的效果，相对于直接预测一个准确的y数值的话。
n_patterns = len(x)
n_vocab = len(chars)

# 把x变成LSTM需要的样子
x = np.reshape(x,(n_patterns,seq_length,1))
# 简单normal到0~1之间
x = x / float(n_vocab)
# output变成one-hot
y = np_utils.to_categorical(y)

print(x[11],y[11])
print(f"x:{x.shape} y:{y.shape}")

from keras.models import load_model
model = load_model(r"E:\algorithm_code\rnn_text_generated\model.hdf5")

# 模型建造  LSTM模型构建
model = Sequential()
model.add(LSTM(128,input_shape=(x.shape[1],x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam')

# 跑模型
# model.fit(x,y,nb_epoch=10,batch_size=128)  #nb_epoch=10,batch_size=32
# model.save(r"E:\algorithm_code\NLP_text_generation\model.hdf5")

# 我们来写个程序，看看我们训练出来的LSTM的效果：
def predict_next(input_array):
    x = np.reshape(input_array,(1,seq_length,1))
    # print("x：",x,x.shape)
    x = x / float(n_vocab)
    # print("x2:",x,x.shape)
    y = model.predict(x)
    # print("y:",y,y.shape)
    return y

def string_to_index(raw_input):
    res = []
    for c in raw_input[(len(raw_input) - seq_length):]:
        res.append(char_to_int[c])
    # print("res:",res)
    res1 = np.array(res)
    return res1

def y_to_char(y):
    largest_index = y.argmax()
    c = int_to_char[largest_index]
    # print("c:",c)
    return c

# 写成一个大程序
def generate_article(init,rounds=10):   #rounds=500
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_char(predict_next(string_to_index(in_string)))
        in_string += n
    # print("in_string:",in_string)
    return in_string

init = 'Professor Michael S. Hart is the originator of the Project.'  #a terrible thing.
init1 = 'The furthest distance in the world,Is not between life and death,But when I stand in front of you,Yet you do not know that I love you'
print(len(init1))
article = generate_article(init1)
print("article:",article)