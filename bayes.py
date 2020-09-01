import re
import numpy as np
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

train_path = 'trains.txt'
test_path = 'tests_text.txt'

p = re.compile(r'<.+?>(.+?)</.+?>')

def remove_tag(line):   # 把文本挑选出来
    m = p.findall(line)
    if len(m) == 1:
        return m[0].strip()
    else:
        return ''

def removePunctuation(text):
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(remove_chars, '', text).strip()
    return text

def get_wordnet(word):  # 获取单词的词性
    if word.startswith('J'):
        return wordnet.ADJ
    elif word.startswith('V'):
        return wordnet.VERB
    elif word.startswith('N'):
        return wordnet.NOUN
    elif word.startswith('R'):
        return wordnet.ADV
    else:
        return None

def read_data(path):
    dataSet = []; label = []; polarity = -1; text = []
    for line in open(path):
        line = line.strip()
        if len(line) > 0:
            if line[:10] == '<Polarity>':
                polarity = int(remove_tag(line))
            if line[:6] == '<text>':
                tokens = removePunctuation(remove_tag(line).lower()).split()
                word_tag = pos_tag(tokens)  # 获得单词词性
                wnl = WordNetLemmatizer()
                lemmas_sent = []
                for tag in word_tag:
                    wordnet_pos = get_wordnet(tag[1]) or wordnet.NOUN
                    lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
                text = np.array([word for word in lemmas_sent if word not in stopwords.words('english')])  # 去除停用词
            if len(text) > 0 and polarity != -1:
                label.append(polarity)
                dataSet.append(text)
                text = []
                polarity = -1
    return np.array(dataSet), np.array(label)

def read_test(path):
    dataSet = []; text = []
    for line in open(path):
        line = line.strip()
        if len(line) > 0:
            if line[:6] == '<text>':
                tokens = removePunctuation(remove_tag(line).lower()).split()
                word_tag = pos_tag(tokens)  # 获得单词词性
                wnl = WordNetLemmatizer()
                lemmas_sent = []
                for tag in word_tag:
                    wordnet_pos = get_wordnet(tag[1]) or wordnet.NOUN
                    lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
                text = np.array([word for word in lemmas_sent if word not in stopwords.words('english')])  # 去除停用词
            if len(text) > 0:
                dataSet.append(text)
                text = []
    return np.array(dataSet)

def train_BOOL(corpus):
    idl = {}  # 标明特征向量的列号
    word_num = {}  # 词汇个数统计，dict格式
    # 统计各词出现个数
    id = 0
    for Corpus in corpus:
        for word in Corpus:
            if word not in word_num:
                word_num[word] = 1
                idl[word] = id
                id += 1
    # 生成矩阵
    X = np.zeros((len(corpus), len(word_num)))
    for i in range(len(corpus)):
        Corpus = corpus[i]
        for j in range(len(Corpus)):
            feature = Corpus[j]
            feature_col = idl[feature]
            X[i, feature_col] = 1
    return X, np.array([i for i in word_num.keys()])

def test_BOOL(corpus, wordlst): # 传入语料和特征单词列表
    # 统计各词出现个数
    word_index = {word: 0 for word in wordlst}  # 列号字典
    for i in range(len(wordlst)):
        word_index[wordlst[i]] = i
    X = np.zeros((len(corpus), len(wordlst))) # 生成矩阵
    for i in range(len(corpus)):  # 每行
        singleCorpus = corpus[i]     # 每行的列表
        for word in singleCorpus:  # 单词
            if word in wordlst:
                feature_col = word_index[word]
                X[i, feature_col] = 1
    return X.astype(int)

def bayes(train_mat, train_class):  # 文档布尔矩阵和标签向量
    Trainlen = len(train_mat)  # 训练文档数目
    Wordlen = len(train_mat[0])   # 总单词数
    pA = sum(train_class)/float(Trainlen)  # 文档属于正类的概率
    # 词条出现数初始化为1,拉普拉斯平滑
    p0Num = np.ones(Wordlen); p1Num = np.ones(Wordlen)
    # 分母初始化为2，拉普拉斯平滑  类别数量为2
    p0D = 2.0; p1D = 2.0
    for i in range(Trainlen):
        # 统计属于正类的条件概率数据，即P(w0|1),P(w1|1),P(w2|1)...
        if train_class[i] == 1:
            p1Num += train_mat[i]  # 统计所有正类中每个单词出现的个数
            p1D += sum(train_mat[i])  # 统计一共出现的正类单词的个数
        # 统计属于负类的条件概率所需的数据
        else:
            p0Num += train_mat[i]
            p0D += sum(train_mat[i])
    p1 = np.log(p1Num / p1D)  # 取对数，防止下溢出  正类的条件概率数组
    p0 = np.log(p0Num / p0D)  # 负类的条件概率数组
    # 返回属于正类的条件概率数组、属于负类的条件概率数组、文档属于正类的概率
    return p0, p1, pA

# 朴素贝叶斯分类器分类
def classify(test_mat, p0, p1, p1_class):
    # 对应元素相乘，logA*B = logA + logB, 这里是累加
    p1 = sum(test_mat * p1) + np.log(p1_class)
    p0 = sum(test_mat * p0) + np.log(1.0 - p1_class)
    if p1 > p0:
        return 1
    else:
        return 0

def test():
    # 创建实验样本
    dataSet, label = read_data(train_path)
    meanTrueRate = 0
    i = 0
    for k in range(5):
        trainData = np.concatenate((dataSet[:i], dataSet[i+200:]), axis=0)
        trainLabel = np.concatenate((label[:i], label[i+200:]), axis=0)
        devData = dataSet[i:i+200]
        devLabel = label[i:i+200]
        train_mat, word = train_BOOL(trainData)
        dev_mat = test_BOOL(devData, word)
        # 创建词汇表,将输入文本中的不重复的单词进行提取组成单词向量
        p0, p1, pA = bayes(train_mat, trainLabel)
        error = 0
        for j in range(dev_mat.shape[0]):
            if classify(dev_mat[j], p0, p1, pA)!= devLabel[j]:
                error += 1
        trueRate = 1 - float(error) / dev_mat.shape[0]
        meanTrueRate += trueRate
        print('第', k+1 ,'次验证正确率为：', trueRate)
        i += 200
    print('平均正确率为:', float(meanTrueRate/5))

    testData = read_test(test_path)
    train_mat, word = train_BOOL(dataSet)
    test_mat = test_BOOL(testData, word)
    p0, p1, pA = bayes(train_mat, label)
    fr = open('texts_prelabel.txt', 'w')
    for j in range(test_mat.shape[0]):
        fr.write(str(classify(test_mat[j], p0, p1, pA)) + '\n')
    fr.close()

test()
