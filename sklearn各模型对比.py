import re
import numpy as np
train_path = 'trains.txt'
test_path = 'tests_text.txt'

p = re.compile(r'<.+?>(.+?)</.+?>')

def remove_tag(line):   # 把文本挑选出来
    m = p.findall(line)
    if len(m) == 1:
        return m[0].strip()
    else:
        return ''

def read_data(path):
    dataSet = []; label = []; polarity = -1; text = []
    for line in open(path):
        line = line.strip()
        #print('line', line)
        if len(line) > 0:
            if line[:10] == '<Polarity>':
                polarity = int(remove_tag(line))
            if line[:6] == '<text>':
                text = remove_tag(line)
            if len(text) > 0 and polarity != -1:
                label.append(polarity)
                dataSet.append(text)
                text = []
                polarity = -1
    return np.array(dataSet), np.array(label)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn.linear_model.logistic import LogisticRegression

# 选取下面的8类
selected_categories = [
    'comp.graphics',
    'rec.motorcycles',
    'rec.sport.baseball',
    'misc.forsale',
    'sci.electronics',
    'sci.med',
    'talk.politics.guns',
    'talk.religion.misc']

# 加载数据集
texts, labels= read_data(train_path)
i = 0
t1 = 0; t2 = 0; t3 = 0; t4 = 0; t5 = 0
for k in range(5):
    train_texts = np.concatenate((texts[:i], texts[i+200:]), axis=0)
    train_labels = np.concatenate((labels[:i], labels[i+200:]), axis=0)
    test_texts = texts[i:i+200]
    test_labels = labels[i:i+200]
    # 贝叶斯
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=2000)),
                   ('clf',MultinomialNB())])
    text_clf=text_clf.fit(train_texts,train_labels)
    predicted=text_clf.predict(test_texts)
    t1+= np.mean(predicted==test_labels)
    print("MultinomialNB准确率为：", np.mean(predicted==test_labels))

    # LogisticRegression
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=2000)),
                   ('clf',LogisticRegression())])
    text_clf=text_clf.fit(train_texts,train_labels)
    predicted=text_clf.predict(test_texts)
    t2+= np.mean(predicted==test_labels)
    print("LogisticRegression准确率为：", np.mean(predicted==test_labels))

    # SVM
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=2000)), ('clf',NuSVC())])
    text_clf=text_clf.fit(train_texts,train_labels)
    predicted=text_clf.predict(test_texts)
    t3+= np.mean(predicted==test_labels)
    print("SVC准确率为：", np.mean(predicted==test_labels))

    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=2000)),
                   ('clf',LinearSVC())])
    text_clf=text_clf.fit(train_texts,train_labels)
    predicted=text_clf.predict(test_texts)
    t4+= np.mean(predicted==test_labels)
    print("LinearSVC准确率为：", np.mean(predicted==test_labels))

    # KNeighborsClassifier
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=2000)),
                   ('clf',KNeighborsClassifier())])
    text_clf=text_clf.fit(train_texts,train_labels)
    predicted=text_clf.predict(test_texts)
    t5 += np.mean(predicted==test_labels)
    print("KNeighborsClassifier准确率为：", np.mean(predicted==test_labels))

    i+=200

print("MultinomialNB准确率为：", t1/5)
print("LogisticRegression准确率为：", t2/5)
print("SVC准确率为：", t3/5)
print("LinearSVC准确率为：", t4/5)
print("KNeighborsClassifier准确率为：", t5/5)