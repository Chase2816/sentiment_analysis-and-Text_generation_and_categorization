# word2vec 训练词向量

import os
import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

import nltk.data
# nltk.download('punkt')
# from nltk.corpus import stopwords

from gensim.models.word2vec import Word2Vec


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
    print('Number of reviews:{}'.format(len(df)))
    return df


# 读入无标签数据   用于训练生成word2vec词向量
df1 = load_dataset(name='unlabeled_train')
df2 = load_dataset(name='labeled_train')
df = pd.concat([df1, df2], axis=0)
print(df.head())  # 查看头部信息
print(df.shape)

# 数据里有"<br /><br />"这些符号，做跟刚刚一样的预处理
# 这里有一点不一样，留了一个筛选，可以去除停用词，也可以不去除停用词
# eng_stopwords = set(stopwords.words('english'))
# 这里用的自定义停用词 ，上面一句是nltk库的停用词
eng_stopwords = {}.fromkeys(
    [line.rstrip() for line in open(r'E:\algorithm_code\english_sentiment_nlp\data\initial_data\stopwords.txt')])


def clean_text(text, remove_stopwords=False):
    text = BeautifulSoup(text, 'html.parser').get_text()  # 解析获取文本信息
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in eng_stopwords]
    return words


# nltk.data.load('tokenizers/punkt/english.pickle')加载英文的划分句子的模型
# 自己写的英文文本，没有出结果，因为在句子结束没有留空格，这是中英文书写习惯造成的
# tokenizers/punkt/ 这里面有好多训练好的模型，只能划分成句子，不能划分成单词
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def print_call_counts(f):  # 统计count
    n = 0

    def wrapped(*args, **kwargs):
        nonlocal n
        n += 1
        if n % 1000 == 1:
            print('method {} called {} times'.format(f.__name__, n))
        return f(*args, **kwargs)

    return wrapped


@print_call_counts
def split_sentences(review):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = [clean_text(s) for s in raw_sentences if s]
    return sentences


sentences = sum(df.review.apply(split_sentences), [])

print('{} review -> {} sentence'.format(len(df), len(sentences)))

# 用gensim训练词嵌入模型
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 设定词向量训练的参数
num_feature = 300  # 词向量的维数
min_word_count = 40  # 最低词数
num_workers = 2  # 4       # 运行的线程数
context = 10  # 上下文窗口大小
downsampling = 1e-3  # 下采样设置频繁的单词  负例采样

model_name = '{}features_{}minwords_{}context.model'.format(num_feature, min_word_count, context)

print("Training model...")
model = Word2Vec(sentences, workers=num_workers,
                 size=num_feature, min_count=min_word_count,
                 window=context, sample=downsampling)
# 如果您不打算进一步培训模型，那么调用init_sims将使模型具有更高的内存效率。
model.init_sims(replace=True)

# 创建一个有意义的模型名和是很有帮助的
# 保存模型供以后使用。您可以稍后使用word2vector .load()加载它
model.save(os.path.join(r'E:\algorithm_code\english_sentiment_nlp\models', model_name))

# 查看训练的词向量结果如何
# “doesnt_match”函数将试图推断出它在一组词是不是来自另外的组:
# 模型可以区分不同的意义!它知道,男人,妇女和儿童比他们更相似的厨房。
# 更多勘探表明,该模型是敏感的细微差别的意义,比如国家和城市之间的差异
print(model.doesnt_match('man women child kitchen'.split()))  # 输出：kitchen
print(model.doesnt_match('france english germany berlin'.split()))  # 输出：berlin

# .most_similar('man') 与man最接近的词 关联度最高   “most_similar”功能来获得洞察集群模型的词:
#  现在训练好了词向量
man = model.most_similar('man')
print(
    man)  # 输出：[('woman', 0.6331404447555542), ('lady', 0.5840568542480469), ('lad', 0.5736382007598877), ('chap', 0.5335173010826111), ('person', 0.533453106880188), ('soldier', 0.5261613726615906), ('men', 0.520458996295929), ('monk', 0.5142411589622498), ('priest', 0.5123844146728516), ('doctor', 0.5079026222229004)]

queen = model.most_similar('queen')
print(
    queen)  # 输出：[('princess', 0.6828762292861938), ('maid', 0.6642452478408813), ('belle', 0.6572738885879517), ('sylvia', 0.6485269069671631), ('dorothy', 0.6425493359565735), ('dietrich', 0.6350311040878296), ('regina', 0.6298528909683228), ('nurse', 0.6266886591911316), ('nina', 0.6253798007965088), ('temple', 0.6239591836929321)]

awful = model.most_similar('awful')
print(
    awful)  # 输出：[('terrible', 0.7525264620780945), ('horrible', 0.7381842136383057), ('atrocious', 0.7341984510421753), ('abysmal', 0.7160285711288452), ('dreadful', 0.708041250705719), ('horrid', 0.676166296005249), ('horrendous', 0.6602495908737183), ('appalling', 0.6442068219184875), ('lousy', 0.6423375606536865), ('amateurish', 0.6356383562088013)]
