#中文情感分析

from sklearn.model_selection import train_test_split #分割数据成训练测试
from gensim.models.word2vec import Word2Vec  #构建词向量
import numpy as np  #数值计算
import pandas as pd  #读取数据
import jieba  #中文分词
from sklearn.externals import joblib  #模型的权重，把数据以二进制丢出来
from sklearn.svm import SVC
import importlib,sys
importlib.reload(sys)


# 载入数据，做预处理(分词)，切分训练集与测试集
def load_file_and_preprocessing():
    neg = pd.read_excel(r'E:\algorithm_code\chinese_sentiment_nlp\data\neg.xls',header=None,index=None)
    pos = pd.read_excel(r'E:\algorithm_code\chinese_sentiment_nlp\data\pos.xls',header=None,index=None)

    cw = lambda x: list(jieba.cut(x)) #对文本分词，组成列表
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)
    print(pos['words'])
    #use 1 for positive sentiment, 0 for negative 正面情绪用1，负面情绪用0
    y = np.concatenate((np.ones(len(pos)),np.zeros(len(neg))))

    x_train,x_test,y_train,y_test = train_test_split(np.concatenate(
        (pos['words'],neg['words'])),y,test_size=0.2)

    np.save(r'E:\algorithm_code\chinese_sentiment_nlp\svm_data/y_train.npy',y_train)
    np.save(r'E:\algorithm_code\chinese_sentiment_nlp\svm_data/y_test.npy',y_test)
    return x_train,x_test

# 对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(text,size,imdb_w2v):
    # 这里所有词向量取平均 也可以去用其他方式
    vec = np.zeros(size).reshape((1,size))
    count = 0
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1,size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

# 计算词向量
def get_train_vecs(x_train,x_test):
    n_dim = 300
    # 初始化模型和词表
    imdb_w2v = Word2Vec(size=n_dim,min_count=10) # 构建一个模型
    imdb_w2v.build_vocab(x_train)

    # 在评论训练集上建模(可能会花费几分钟)
    imdb_w2v.train(x_train,total_examples=len(x_train),epochs=1)  #len(x_train)=16884  x_train.shape=(16884,0)

    train_vecs = np.concatenate([build_sentence_vector(
        z,n_dim,imdb_w2v) for z in x_train])
    # train_vecs = scale(train_vecs)

    np.save(r'E:\algorithm_code\chinese_sentiment_nlp\svm_data/train_vecs.npy',train_vecs)
    print(train_vecs.shape)

    # 在测试集上训练
    imdb_w2v.train(x_test,total_examples=len(x_test),epochs=1)
    imdb_w2v.save(r'E:\algorithm_code\chinese_sentiment_nlp\svm_data\w2v_model/w2v_model.pkl')
    # Build test tweet vectors then scale构建测试tweet向量，然后进行缩放
    test_vecs = np.concatenate([build_sentence_vector(
        z,n_dim,imdb_w2v) for z in x_test])
    # test_vecs = scale(test_vecs)
    np.save(r'E:\algorithm_code\chinese_sentiment_nlp\svm_data/test_vecs.npy',test_vecs)
    print(test_vecs.shape)

# 获取数据
def get_data():
    train_vecs = np.load(r'E:\algorithm_code\chinese_sentiment_nlp\svm_data/train_vecs.npy')
    y_train = np.load(r'E:\algorithm_code\chinese_sentiment_nlp\svm_data/y_train.npy')
    test_vecs = np.load(r'E:\algorithm_code\chinese_sentiment_nlp\svm_data/test_vecs.npy')
    y_test = np.load(r'E:\algorithm_code\chinese_sentiment_nlp\svm_data/y_test.npy')
    return train_vecs, y_train, test_vecs, y_test

# 训练svm模型
def svm_train(train_vecs,y_train,test_vecs,y_test):
    clf = SVC(kernel='rbf',verbose=True)
    clf.fit(train_vecs,y_train)
    joblib.dump(clf,r'E:\algorithm_code\chinese_sentiment_nlp\svm_data\svm_model/model.pkl')
    print(clf.score(test_vecs,y_test))

# 构建待预测句子的向量
def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load(r'E:\algorithm_code\chinese_sentiment_nlp\svm_data\w2v_model/w2v_model.pkl')
    #imdb_w2v = train(words)
    train_vecs = build_sentence_vector(words,n_dim,imdb_w2v)
    print(train_vecs.shape)
    return train_vecs

# 对单个句子进行情感分类
def svm_predict(string):
    words = jieba.lcut(string)
    words_vecs = get_predict_vecs(words)
    clf = joblib.load(r'E:\algorithm_code\chinese_sentiment_nlp\svm_data\svm_model\model.pkl')

    result = clf.predict(words_vecs)

    if int(result[0]) == 1:
        print(string,'positive')
    else:
        print(string,'negative')

if __name__ == '__main__':
    # x_train, x_test = load_file_and_preprocessing()
    # print(x_train,x_train.shape,len(x_train))
    # print(x_test.shape)
    # get_train_vecs(x_train,x_test)

    # train_vecs, y_train, test_vecs, y_test = get_data()
    # svm_train(train_vecs, y_train, test_vecs, y_test)


    # 对输入句子情感进行判断
    string = '电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    svm_predict(string)
    string='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    svm_predict(string)

# 小练习  用gensim的doc2vec和LSTM神经网络分类  效果比SVM更好