# 用每日新闻预测金融市场变化（进阶版）
# 使用FastText来做分类

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import datetime

# 一。监视数据
# 读入数据
data = pd.read_csv(r"E:\algorithm_code\NLP_text_classification\data\Combined_News_DJIA.csv")

# 查看数据  Label如果是1，那么当日的DJIA就提高或者不变了。如果是0，那么DJIA那天就是跌了。
print(data.head())

# 二。分割测试/训练集
print("【1.1】正在划分训练/测试集合...")
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-12']

# 把每条新闻做成一个单独的句子，集合在一起
X_train = train[train.columns[2:]]  # columns列 train.columns[2:]为train数据从第三列开始的特征名
print("【1.2】正在获取语料库flatten,corpus...")
# !corpus(语料库)是全部我们『可见』的文本资料。我们假设每条新闻就是一句话，
# 把他们全部flatten()了，我们就会得到list of sentences。
# 同时我们的X_train和X_test可不能随便flatten，他们需要与y_train和y_test对应。
# flatten()函数将一个嵌套多层的数组array转化成只有一层的数组
corpus = X_train.values.flatten().astype(str)

X_train = X_train.values.astype(str)
X_train = np.array([' '.join(x) for x in X_train])
X_test = test[test.columns[2:]]
X_test = X_test.values.astype(str)
X_test = np.array([' '.join(x) for x in X_test])

y_train = train['Label'].values
y_test = test['Label'].values

# 这里注意，需要三样东西：
# corpus是全部我们『可见』的文本资料。
# 我们假设每条新闻就是一句话，把他们全部flatten()了，我们就会得到list of sentences。
# 同时我们的X_train和X_test可不能随便flatten，他们需要与y_train和y_test对应。

print("corpus[:3]:", corpus[:3])
print("X_train[:1]:", X_train[:1])
print("y_train[:5]:", y_train[:5])

# 把每个单词给分隔开：
# 同样，corpus和X_train的处理不同
from nltk.tokenize import word_tokenize

corpus = [word_tokenize(x) for x in corpus]
X_train = [word_tokenize(x) for x in X_train]
X_test = [word_tokenize(x) for x in X_test]

# tokenize完毕后，
# 我们可以看到，虽然corpus和x都是一个二维数组，但是他们的意义不同了。
# corpus里，第二维数据是一个个句子。
# x里，第二维数据是一个个数据点（对应每个label）
print("X_train[:2]:", X_train[:2])
print("corpus[:2]:", corpus[:2])

# 三。预处理
# 进行一些预处理来把我们的文本资料变得更加统一：
# 1.小写化 2.删除停止词  3.删除数字与符号  4.lemma
# 我们把这些功能合为一个func

# 停止词
from nltk.corpus import stopwords

stop = stopwords.words('english')

# 数字
import re


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


# 特殊符号
def isSymbol(inputString):
    return bool(re.search(r'[^\w]', inputString))


# lemma 词形归一
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
        if check(word):  # 这一段的用处仅仅是去除python里面byte存str时候留下的标识。。之前数据没处理好，其他case里不会有这个情况
            word = word.lower().replace("b'", '').replace('b"', '').replace('"', '').replace("'", '')
            res.append(wordnet_lemmatizer.lemmatize(word))
    return res


#  把我们三个数据组都来处理一下：
corpus = [preprocessing(x) for x in corpus]
X_train = [preprocessing(x) for x in X_train]
X_test = [preprocessing(x) for x in X_test]

# 看看处理之后的数据长相：
print("corpus[553]", corpus[553])
print("X_train[523]", X_train[523])

# 四。训练NLP模型
# 有了这些干净的数据集，我们可以做我们的NLP模型了。
# 我们这里要用的是FastText。
# 原理，我在课件上已经讲过了，这里我们来进一步看看具体的使用。
# 由于这篇paper刚刚发布，很多社区贡献者也都在给社区提供代码，尽早实现python版本的开源编译（我也是其中之一）。
# 当然，因为Facebook团队本身已经在GitHub上放出了源代码（C++），
# 所以，我们可以用一个python wrapper来造个interface，方便我们调用。
# 首先，我们讲过，FT把label也看做一个元素，带进了word2vec的网络中。
# 那么，我们就需要把这个label塞进我们的“句子”中

for i in range(len(y_train)):
    label = '__label__' + str(y_train[i])
    X_train[i].append(label)
print("X_train[49]", X_train[49])

# 然后，我们把数据存成文件的形式。因为我们这里的FastText只是个python的interface。调用起来还得用C++的接口。
# 我们需要存三个东西：
# 1.含有label的train集
# 2.不含label的test集
# 3.label单独放一个文件
X_train = [' '.join(x) for x in X_train]
print("X_train[12]", X_train[12])

# 同理，test集也这样。
X_test = [' '.join(x) for x in X_test]

with open(r"E:\algorithm_code\NLP_text_classification\data/train_ft.txt", 'w') as f:
    for sen in X_train:
        f.write(sen + '\n')

with open(r"E:\algorithm_code\NLP_text_classification\data/test_ft.txt", 'w') as f:
    for sen in X_test:
        f.write(sen + '\n')

with open(r"E:\algorithm_code\NLP_text_classification\data/test_label_ft.txt", 'w') as f:
    for label in y_test:
        f.write(str(label) + '\n')

# 五。调用FastText模块
import fasttext  #pip install fasttext-win

clf = fasttext.supervised(r"E:\algorithm_code\NLP_text_classification\data/train_ft.txt", 'model', dim=256, ws=5, neg=5,
                          epoch=100, min_count=10, lr=0.1, lr_update_rate=1000, bucket=200000)

# 训练完我们的FT模型后，我们可以测试我们的Test集了
y_scores = []

# 我们用predict来给出判断
labels = clf.predict(X_test)
print(labels)
print(y_test)
y_preds = np.array(labels).flatten().astype(int)
print(y_preds.shape)
print(y_test.shape)
# 我们来看看
print("y_test:", len(y_test))
print("y_test:", y_test)
print("y_preds:", len(y_preds))
print("y_preds:", y_preds)

from sklearn import metrics

# 算个AUC准确率
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_preds, pos_label=1)
print("AUC:", metrics.auc(fpr, tpr))

# 同理，这里，我们通过parameter tuning或者是resampling，可以让我们的结果更加好。
# 当然，因为FT本身也是一个word2vec。并且自带了一个类似于二叉树的分类器在后面。
# 这样，在小量数据上，是跑不出很理想的结论的，还不如我们自己带上一个SVM的效果。
# 但是面对大量数据和大量label，它的效果就体现出来了。
