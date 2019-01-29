import os
import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

from nltk.corpus import stopwords  # 停止词

from gensim.models.word2vec import Word2Vec

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans


def load_dataset(name, nrows=None):
    dataset = {
        'unlabeled_train': 'unlabeledTrainData.tsv',
        'labeled_train': 'labeledTrainData.tsv',
        'test': 'testData.tsv'
    }
    if name not in dataset:
        raise ValueError(name)
    data_file = os.path.join(r'E:\algorithm_code\english_sentiment_nlp\data\initial_data', dataset[name])
    df = pd.read_csv(data_file, sep='\t', escapechar='\\', nrows=nrows)
    print('Number of review: {}'.format(len(df)))
    return df


eng_stopwords = set(stopwords.words('english'))


def clean_text(text, remove_stopwords=False):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in eng_stopwords]
    return words


### 读入之前训练好的Word2Vec模型
model_name = '300features_40minwords_10context.model'
model = Word2Vec.load(os.path.join(r'E:\algorithm_code\english_sentiment_nlp\models', model_name))

# 根据word2vec的结果去对影评文本进行编码
# 编码方式有一点粗暴，简单来说就是把这句话中的词的词向量做平均
df = load_dataset('labeled_train')
print(df.head())  # data:25000


def to_review_voctor(review):  # 转成评论数据向量的函数
    # 清洗数据 clean_text（）
    words = clean_text(review, remove_stopwords=True)
    # model的每句话由很多个词向量组成的np.array矩阵
    array = np.array([model[w] for w in words if w in model])
    return pd.Series(array.mean(axis=0))


# .apply（to_review_voctor）函数把所有文本都过一次
train_data_features = df.review.apply(to_review_voctor)
print(train_data_features.head())

# 用随机森林构建分类器  也可以用别的   n_estimators=100 100棵树
forest = RandomForestClassifier(n_estimators=100, random_state=42)
# .fit()在现在的数据集上做个训练
forest = forest.fit(train_data_features, df.sentiment)

# 同样在训练集上测试，确保模型能正常work
confusion_matrix(df.sentiment, forest.predict(train_data_features))

# 清理占用内容的变量
del df
del train_data_features

# 预测测试集结果并上传kaggle
df = load_dataset('test')
print(df.head())  # data:25000

test_data_features = df.review.apply(to_review_voctor)
print(test_data_features.head())

result = forest.predict(test_data_features)  # 做预测
output = pd.DataFrame({'id': df.id, 'sentiment': result})
output.to_csv(os.path.join(r'E:\algorithm_code\english_sentiment_nlp\data', 'Word2vec_mode.csv'), index=False)
print(output.head())

del df
del test_data_features
del forest

# 对词向量进行聚类研究和编码  使用Kmeans进行聚类
word_vectors = model.wv.syn0
print(word_vectors)
print(word_vectors.shape)
num_clusters = word_vectors.shape[0] // 10  # 10个样本在一起

kmeans_clustering = KMeans(n_clusters=num_clusters, n_jobs=1)  # n_jobs=4
idx = kmeans_clustering.fit_predict(word_vectors)

word_centroid_map = dict(zip(model.wv.index2word, idx))

import pickle

filename = 'word_centroid_map_10avg.pickle'
with open(os.path.join('models', filename), 'bw') as f:
    pickle.dump(word_centroid_map, f)

# with open(os.path.join('models',filename),'br') as f:
#     word_centroid_map = pickle.load(f)

# 输出一些clusters看
for cluster in range(0, 10):
    print("\nCluster %d" % cluster)
    print([w for w, c in word_centroid_map.items() if c == cluster])

# 把评论数据转成cluster bag vectors
wordset = set(word_centroid_map.keys())


def make_cluster_bag(review):
    words = clean_text(review, remove_stopwords=True)
    return (
        pd.Series([word_centroid_map[w] for w in words if w in wordset]).value_counts().reindex(range(num_clusters + 1),
                                                                                                fill_value=0))


df = load_dataset('labeled_train')
print(df.head())

train_data_features = df.review.apply(make_cluster_bag)
print(train_data_features.head())

# 再用随机森林算法建模
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest = forest.fit(train_data_features, df.sentiment)

# 在训练集上测试，确保模型能正常work
confusion_matrix(df.sentiment, forest.predict(train_data_features))

# 去掉无用的占内存的量
del df
del train_data_features

# 载入测试数据做预测
df = load_dataset('test')
print(df.head())  # data:25000

test_data_features = df.review.apply(make_cluster_bag)
print(test_data_features.head())

result = forest.predict(test_data_features)
output = pd.DataFrame({'id': df.id, 'sentiment': result})
output.to_csv(os.path.join('data', 'Word2Vec_BagOfClusters.csv'), index=False)
print(output)
print(output.head())

del df
del test_data_features
del forest
