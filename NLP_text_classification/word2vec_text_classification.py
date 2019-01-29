# 用每日新闻预测金融市场变化（进阶版）  CNN4Text
# Kaggle竞赛：https://www.kaggle.com/aaron7sun/stocknews
# 学习如何更有逼格地使用word2vec

import tensorflow as tf
print(tf.__version__)

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import datetime

# 监视数据
# 先读入数据
data = pd.read_csv(r"E:\algorithm_code\NLP_text_classification\data\Combined_News_DJIA.csv")

# 查看数据样子 。如果是1，那么当日的DJIA就提高或者不变了。如果是0，那么DJIA那天就是跌了。
print("data:", data.head())

# 分割测试/训练集
# 先把数据给分成Training/Testing data
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

# 然后，把每条新闻做成有一个单独的句子，集合在一家
X_train = train[train.columns[2:]]
corpus = X_train.values.flatten().astype(str)

X_train = X_train.values.astype(str)
X_train = np.array([' '.join(x) for x in X_train])
X_test = test[test.columns[2:]]
X_test = X_test.values.astype(str)
X_test = np.array([' '.join(x) for x in X_test])
y_train = train['Label'].values
y_test = test['Label'].values
print(f"y_train:{y_train.shape}")
print(f"y_test:{y_test.shape}")

# 注意：这里需要三样东西
# corpus是全部我们’可见‘的文本资料。我们假设每条新闻就是一句话，
# 把他们全部flatten()了，我们就会得到list of sentences。
# 同时我们的X_train和X_test可不能随便flatten，他们需要与y_train和y_test
print("corpus[:3]:", corpus[:3])
print(f"X_train[:1]:{X_train[:1]}")
print(f"y_train[:5]:{y_train[:5]}")

# 再把每个单词给分隔开：同样，corpus和X_train的处理
from nltk.tokenize import word_tokenize

corpus = [word_tokenize(x) for x in corpus]
X_train = [word_tokenize(x) for x in X_train]
X_test = [word_tokenize(x) for x in X_test]
# tokenize完毕后，可以看到，虽然corpus和x都是一个二维数组，但是他们的意义不同了。
# corpus里，第二维数据是一个个句子。
# x里，第二维数据是一个个数据点（对应每个label）
print("X_train[:2]:", X_train[:2])
print(f"corpus[:2]:{corpus[:2]}")

# 预处理
# 进行一些预处理来把文本资料变得更加统一：
# 小写化；删除停止词；删除数字与符号；lemma 。把这些功能合为一个func：
# 停止词
from nltk.corpus import stopwords

stop = stopwords.words('english')

# 数字
import re


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


# 特殊符号
def isSymbol(inputString):
    return bool(re.match(r'[^\w]', inputString))


# lemma
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()


def check(word):
    """
    如果需要这个单词，则True
    如果应该去除，则False
    """
    word = word.lower()
    if word in stop:
        return False
    elif hasNumbers(word) or isSymbol(word):
        return False
    else:
        return True


# 把上面的方法综合起来
def preprocessing(sen):
    res = []
    for word in sen:
        if check(word):
            # 这一段的用处仅仅是去除python里面byte存str时候留下的标识。。
            # 之前数据没处理好，其他case（案例）里不会有这个情况
            word = word.lower().replace("b'", '').replace('b"', '').replace('"', '').replace("'", '')
            res.append(wordnet_lemmatizer.lemmatize(word))
    return res


# 把三个数据组都来处理一下
corpus = [preprocessing(x) for x in corpus]
X_train = [preprocessing(x) for x in X_train]
X_test = [preprocessing(x) for x in X_test]

# 查看处理后的数据长相
print(f"corpus[553]:{corpus[553]}")
print(f"X_train[523]:{X_train[523]}")

# 训练NLP模型
# 用word2vec
from gensim.models.word2vec import Word2Vec

model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4)
# 这时候，每个单词都可以像查找字典一样，读出他们的w2v坐标了：
print(f"model['ok']{model['ok']}")

# 用NLP模型表达我们的X
# 用这个坐标，来表示我们之前干干净净的X，
# 这里有个问题，我么的Vec是基于每个单词的，怎么办呢？
# 由于我们文本本身的量很小，可以把所有的单词的vector拿过来取个平均值

# 先拿到全部的vocabulary
vocab = model.wv.vocab


# 得到任意text的vector
def get_vector(word_list):
    # 建立一个全是0的array
    res = np.zeros([128])
    count = 0
    for word in word_list:
        if word in vocab:
            res += model[word]
            count += 1
    return res / count


# 得到一个任意word list平均vector值得方法：
print(f"get_vector:{get_vector(['hello', 'from', 'the', 'other', 'side'])}")

# 同步把X都转化成128维的一个vector list
# 为了之后内容的方便，先把之前我们处理好的wordlist给存下来。
wordlist_train = X_train
wordlist_test = X_test

X_train = [get_vector(x) for x in X_train]
X_test = [get_vector(x) for x in X_test]

print(f"X_train[10]:{X_train[10]}")

# 建立ML模型
# 这里，128维的每一个值都是连续关系的。不是分裂开考虑的。所以，道理上讲，
# 我们是不太适合用RandomForest（随机森林）这类把每个column（列）当做单独的variable（变量）来看的方法。
# (当然，事实是，你也可以这么用)

# 比较适合连续函数的方法：SVM
# from sklearn.svm import SVR
# from sklearn.model_selection import cross_val_score
#
# params = [0.1, 0.5, 1, 3, 5, 7, 10, 12, 16, 20, 25, 30, 35, 40]
# test_scores = []
# for param in params:
#     clf = SVR(gamma=param)
#     test_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
#     test_scores.append(np.mean(test_scores))

# import matplotlib.pyplot as plt
#
# plt.plot(params, test_scores)
# plt.title("Param vs CV AUC Score")
# plt.show()


# 用CNN来提升逼格
# 你也许会说，这也太扯了吧，直接平均了一下vec值，就能拿来跑？
# 哈哈，当然不会这么简单。
# 必然是有更严谨的处理方式。
# 比如：
# 用vector表示出一个大matrix，并用CNN做“降维+注意力”
# （为了演示的方便，下面内容我会把整个case搞得简单点。要是想更加复杂准确的话，直接调整参数，往大了调，就行）
# 首先，确定一个padding_size。
# 什么是padding size？
# 就是为了让我们生成的matrix是一样的size啊。。长度
# （这里其实可以最简单地调用keras的sequence方法来做，但是我想让大家更清楚的明白一下，内部发生了什么）

# 说明，对于每天的新闻，我们会考虑前256个单词。不够的我们用[000000]补上
# vec_size 指的是我们本身vector的size
def transform_to_matrix(x, padding_size=256, vec_size=128):
    res = []
    for sen in x:
        matrix = []
        for i in range(padding_size):
            try:
                matrix.append(model[sen[i]].tolist())
            except:
                # 这里有两种except情况
                # 1.这个单词找不到
                # 2.sen没这么长
                # 不管哪种情况，我们直接贴上全是0的vec
                matrix.append([0] * vec_size)
        res.append(matrix)
    return res

# 把我们原本的word list跑一遍
X_train = transform_to_matrix(wordlist_train)
X_test = transform_to_matrix(wordlist_test)

print("X_train[123]:",X_train[123])

# 可以看到，现在我们得到的就是一个大大的Matrix，它的size是 128 * 256
# 每一个这样的matrix，就是对应了我们每一个数据点
# 在进行下一步之前，我们把我们的input要reshape一下。
# 原因是我们要让每一个matrix外部“包裹”一层维度。来告诉我们的CNN model，我们的每个数据点都是独立的。之间木有前后关系。
# （其实对于股票这个case，前后关系是存在的。这里不想深究太多这个问题。有兴趣的同学可以谷歌CNN+LSTM这种强大带记忆的深度学习模型。）

# 画acc和loss曲线图   写一个LossHistory类，保存loss和acc
import keras
import matplotlib.pyplot as plt
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        #创建一个图
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')#plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)#设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')#给x，y轴加注释
        plt.legend(loc="upper right")#设置图例显示位置
        plt.show()

# 可以看到，现在我们得到的就是一个大大的Matrix，它的size是 128 * 256
# 每一个这样的matrix，就是对应了我们每一个数据点
# 在进行下一步之前，我们把我们的input要reshape一下。
# 原因是我们要让每一个matrix外部“包裹”一层维度。来告诉我们的CNN model，我们的每个数据点都是独立的。之间木有前后关系。
# （其实对于股票这个case，前后关系是存在的。这里不想深究太多这个问题。有兴趣的同学可以谷歌CNN+LSTM这种强大带记忆的深度学习模型。）
# 搞成np的数组，便于处理
X_train = np.array(X_train)
X_test = np.array(X_test)

# 看看数组的大小
print(f"X_train:{X_train.shape}")  #(1611, 256, 128)
print(f"X_test:{X_test.shape}")    #(378, 256, 128)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
print(f"X_train:{X_train.shape}")  #(1611, 256, 128, 1)
print(f"X_test:{X_test.shape}")    #(378, 256, 128, 1)

# 接下来，我们安安稳稳的定义我们的CNN模型
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers.core import Dense,Dropout,Activation,Flatten

# set parameters:
batch_size = 32
n_filter = 16
filter_length = 3   # 4
nb_epoch = 5
n_pool = 2

# 新建一个sequential的模型   X_train:(1611, 1, 256, 128)  X_test:(378, 1, 256, 128)
model = Sequential()
model.add(Conv2D(n_filter,(filter_length,filter_length),activation='relu',input_shape=(256,128,1)))
model.add(Conv2D(n_filter,(filter_length,filter_length),activation='relu'))
model.add(MaxPooling2D(pool_size=(n_pool,n_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
# 后面接上一个ANN
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))
model.summary()
# compile模型   #编译模型
model.compile(loss='mse',optimizer='adadelta',metrics=['accuracy'])

#创建一个实例LossHistory
history = LossHistory()

# 开始训练  迭代训练（注意这个地方要加入callbacks）
model.fit(X_train,y_train,batch_size=batch_size,epochs=nb_epoch,verbose=0,callbacks=[history])
#模型评估
score = model.evaluate(X_test,y_test,verbose=1)
print(f"Test score：{score[0]}")
print(f"Test accuracy:{score[1]}")
#绘制acc-loss曲线
history.loss_plot('epoch')


# 思考：
# 虽然我们这里使用了word2vec，但是对CNN而言，管你3721的input是什么，只要符合规矩，它都可以process。
# 这其实就给了我们更多的“发散”空间：
# 我们可以用ASCII码（0，256）来表达每个位置上的字符，并组合成一个大大的matrix。
# 这样牛逼之处在于，你都不需要用preprocessing了，因为每一个字符的可以被表示出来，并且都有意义。
# 另外，你也可以使用不同的分类器：
# 我们这里使用了最简单的神经网络~
# 你可以用LSTM或者RNN等接在CNN的那句Flatten之后



