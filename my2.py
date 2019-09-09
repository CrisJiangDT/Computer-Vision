import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network.multilayer_perceptron import MLPClassifier


# 选取下面的8类
'''
selected_categories = [
    'comp.graphics',
    'rec.motorcycles',
    'rec.sport.baseball',
    'misc.forsale',
    'sci.electronics',
    'sci.med',
    'talk.politics.guns',
    'talk.religion.misc']
    '''

selected_categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

# 加载数据集
newsgroups_train=fetch_20newsgroups(subset='train',
                                    categories=selected_categories,
                                    remove=('headers','footers','quotes'))
newsgroups_test=fetch_20newsgroups(subset='train',
                                    categories=selected_categories,
                                    remove=('headers','footers','quotes'))

train_texts=newsgroups_train['data']
train_labels=newsgroups_train['target']
test_texts=newsgroups_test['data']
test_labels=newsgroups_test['target']
print(len(train_texts),len(test_texts))


# MLPClassifier
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',MLPClassifier())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
print("准确率为：",np.mean(predicted==test_labels))

