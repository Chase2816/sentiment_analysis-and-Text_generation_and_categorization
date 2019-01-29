import os
import re
import numpy as np  # np做计算
import pandas as pd  # 读取csv成excel表格

from bs4 import BeautifulSoup  # 解析网页

from sklearn.feature_extraction.text import CountVectorizer  # 转换向量计数
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
from sklearn.metrics import confusion_matrix  # 评估准则

import nltk
# nltk.download()
from nltk.corpus import stopwords

# 用pandas读入训练数据
# pands读取csv文件以excel打开，是一列一列的。每一列是一个维度的信息，每一行表示一个样本
datafile = os.path.join(r'E:\algorithm_code\english_sentiment_nlp\data\initial_data', 'labeledTrainData.tsv')
# datafile = r'G:\PycharmProjects\algorithm_code\data\labeledTrainData.tsv'
df = pd.read_csv(datafile, sep='\t', escapechar='\\')  # '\\'转义 	'\t'横向制表符
print('Number of reviews:{}'.format(len(df)))  # 25000数据
print(df.head())  # 打印图表
print(df['review'][0])  # 查看评论数据


def display(text, title):
    # 用来看数据  时刻对数据要了解
    print(title)
    print('\n------我是分割线-------\n')
    print(text)


raw_example = df['review'][1]
display(raw_example, '原始数据')

example = BeautifulSoup(raw_example, 'html.parser').get_text()
display(example, '去掉HTML标签的数据')

# re.sub（）返回替换最左边得到的字符串
example_letters = re.sub(r"[^a-zA-Z]", ' ', example)  # 把符号换成空格
display(example_letters, '去掉标点的数据')

words = example_letters.lower().split()  # 全部转成小写，用空格分开
display(words, '纯词列表数据')

# 下载停用词和其他语料会用到
# nltk.download('stopwords')
# words_nostop = [w for w in words if w not in stopwords.words('english')]
# 自己搞的停用词stopwords.txt，也可以用NLTK的stopwords
stopword = {}.fromkeys(
    [line.rstrip() for line in open(r"E:\algorithm_code\english_sentiment_nlp\data\initial_data\stopwords.txt")])
print(f"stopword:{stopword}")  # 停用词
words_nostop = [w for w in words if w not in stopword]
display(words_nostop, '去掉停用词数据')

print(f"把上面的函数，定义到一个总函数里")
# eng_stopwords = set(stopwords.words('english'))  #这是nltk的停用词
eng_stopwords = set(stopword)  # 这是自定义的


# print(eng_stopwords)

def clean_text(text):
    # 读取文本 html解析 拿出具体内容
    text = BeautifulSoup(text, 'html.parser').get_text()  # 解析获取文本信息
    # 用正则表达式 把非字母的符号用空格替换掉
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # 把单词转换成小写，做空格分割
    words = text.lower().split()
    # 重新组词
    words = [w for w in words if w not in eng_stopwords]
    return ' '.join(words)


data = clean_text(raw_example)
print(data)

# 把清洗数据添加到dataframe(计算机)里
df['clean_review'] = df.review.apply(clean_text)
print(df.head())  # 查看数据
print("ssds", df.review)  # 是datafile数据文件里面的review数据

# 抽取bag of words(词袋模型)特征(用sklearn的CountVectorizer)
# CountVectorizer是属于常见的特征数值计算类，是一个文本特征提取方法。
# 对于每一个训练文本，它只考虑每种词汇在该训练文本中出现的频率。
vectorizer = CountVectorizer(max_features=5000)
# max_features：默认为None，可设为int，对所有关键词的term frequency进行降序排序，只取前max_features个作为关键词集
# CountVectorizer会将文本中的词语转换为词频矩阵，它通过fit_transform函数计算各个词语出现的次数。
train_data_features = vectorizer.fit_transform(df.clean_review).toarray()  # .toarray()转换成数组
# print(train_data_features.shape) #(25000,5000)
# print(train_data_features)


# 训练分类器
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, df.sentiment)

# 在训练集上做个perdict看看效果如何
perdict = confusion_matrix(df.sentiment, forest.predict(train_data_features))
print(perdict)

# 删除不用的占内容变量
del df
del train_data_features

# 读取测试数据进行预测
datafile = os.path.join(r'E:\algorithm_code\english_sentiment_nlp\data\initial_data', 'testData.tsv')
df = pd.read_csv(datafile, sep='\t', escapechar='\\')
print('Number os review:{}'.format(len(df)))
df['clean_review'] = df.review.apply(clean_text)
print(df.head())

# 根据词典去编一个bag of words 把文本转成词袋模型
test_data_features = vectorizer.transform(df.clean_review).toarray()
print(test_data_features.shape)
print(f'test_data_features:{test_data_features}')

result = forest.predict(test_data_features)
output = pd.DataFrame({'id': df.id, 'sentiment': result})
print(output.head())

output.to_csv(os.path.join(r'E:\algorithm_code\english_sentiment_nlp\data', 'BAG_of_Words_mode.csv'), index=False)
del df
del test_data_features
