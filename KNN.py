import re
import numpy as np
import operator
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

def read_data(path):
    dataSet = []; label = []; polarity = -1; text = []
    for line in open(path):
        line = line.strip()
        #print('line', line)
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
            # if line[:10] == '<u_review>' and flag > 2 and flag < 7:
            # u_review.append(lst); flag += 1
            # if line[:10] == '<b_review>'and flag > 6 and flag < 11:
            # b_review.append(lst); flag += 1
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
            # if line[:10] == '<u_review>' and flag > 2 and flag < 7:
            # u_review.append(lst); flag += 1
            # if line[:10] == '<b_review>'and flag > 6 and flag < 11:
            # b_review.append(lst); flag += 1
            if len(text) > 0:
                dataSet.append(text)
                text = []
    return np.array(dataSet)

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

def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = vector_a * vector_b.T
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)  # l2范数相乘
    cos = float(num / denom)
    return 1- cos   # 便于后面不要反序

def pearsonr(x, y):
    return (((x - x.mean())/(x.std(ddof = 0)))*((y - y.mean())/(y.std(ddof = 0)))).mean()

def pearsonr_select(data_matrix, data_class, threshold):  # 皮尔森特征选择
    corr = np.zeros((1, data_matrix.shape[1]))
    for i in range(data_matrix.shape[1]):
        corr[0, i] = pearsonr(data_matrix[:, i], data_class)
    corr = np.fabs(corr[0])  # 取绝对值
    mu = corr > threshold
    data_matrix = data_matrix[:, mu]
    return data_matrix, mu

def KNN(test_data, train_matrix, train_class, k):
    size = train_matrix.shape[0]
    #distances = (((np.tile(test_data, (size,1)) - train_matrix)**2).sum(axis = 1))**0.5   # 欧氏距离
    distances = []
    for i in range(size):
        distances.append(cos_sim(test_data, train_matrix[i]))
    distances = np.array(distances)  # 余弦相似度数组
    sortedDist = distances.argsort()  # 排序
    classCount = {}
    for i in range(k):
        label = train_class[sortedDist[i]]
        classCount[label] = classCount.get(label,0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def Test(train_matrix, test_data, train_class, test_class, k): # 传入测试样本特征向量 和标签类别
    errorCount = 0
    Test_num = len(test_class)
    for i in range(Test_num):
        classifyResult = KNN(test_data[i,:], train_matrix, train_class, k)  # 列的维度要一样
        #print(classifierResult)
        if int(classifyResult) != int(test_class[i]):
            errorCount += 1
    trueRate = (Test_num-errorCount)/Test_num
    return trueRate

def testingKNN(threshold, k):
    # 创建实验样本
    dataSet, label = read_data(train_path)
    meanTrueRate = 0
    i = 0
    for m in range(5):
        trainData = np.concatenate((dataSet[:i], dataSet[i+200:]), axis=0)
        trainLabel = np.concatenate((label[:i], label[i+200:]), axis=0)
        devData = dataSet[i:i+200]
        devLabel = label[i:i+200]
        train_mat, word = train_BOOL(trainData)
        # 创建词汇表,将输入文本中的不重复的单词进行提取组成单词向量
        train_matrix, mu = pearsonr_select(train_mat, trainLabel, threshold)
        wordlst = word[mu]
        dev_matrix = test_BOOL(devData, wordlst)
        true_rate = Test(train_matrix, dev_matrix, trainLabel, devLabel, k)
        meanTrueRate += true_rate
        print('第', m + 1, '次验证正确率为：', true_rate)
        i += 200
    print('平均正确率为:', float(meanTrueRate/5))

    #testData = read_test(test_path)
    #train_mat, word = train_BOOL(dataSet)
    #train_matrix, mu = pearsonr_select(train_mat, label, 0.01)
    #wordlst = word[mu]
    #test_mat = test_BOOL(testData, wordlst)
    #fr = open('knn.txt', 'w')
    #for j in range(test_mat.shape[0]):
    #    fr.write(str(KNN(test_mat[j], train_matrix, label, k)) + '\n')
    #fr.close()

testingKNN(0.015, 13)
