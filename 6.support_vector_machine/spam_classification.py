# coding=utf-8
'''
    垃圾邮件分类
    （正则学习）https://docs.python.org/zh-cn/3/library/re.html#regular-expression-syntax
'''
import numpy as np
import re
from nltk.stem import PorterStemmer
from scipy.io import loadmat
from sklearn.svm import SVC


def load_data():
    fe = open('data/emailSample1.txt')
    datas = fe.read()
    return datas


def process_email(email_content):
    '''找到篇章在字典位置的映射'''
    # Lower case
    email_contents = email_content.lower()

    # Stripping HTML,删除html标签
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub(r'[>]+', '', email_contents)

    # Normalizing Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub(r'[0-9]+', 'number', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    # \s：匹配所有空格
    email_contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Normalizing Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Normalizing Dollars
    email_contents = re.sub(r'[$]+', 'dollar', email_contents)

    # Word Stemming
    # Removal of non-words
    email_contents = re.sub(r'[\s]+', ' ', email_contents)

    # 分词
    partition_text = re.split(r'[ @$/#.-:&*+=\[\]?!(){},\'\">_<;%\n\f]', email_contents)
    # Word Indices for Sample Email
    word_indices = []
    vocab = np.loadtxt('data/vocab.txt', dtype=str)
    vocab_list = vocab[:, 1]
    stemmer = PorterStemmer()  # 英文词干提取算法
    for word in partition_text:
        if word != '':
            # 去掉所有的标点符号,非数字、字母字符
            word = re.sub(r'[^a-zA-Z0-9]', '', word)
            # Stem the word
            # (the porterStemmer sometimes has issues, so we use a try catch block)
            word = stemmer.stem(word)
            if word == '':
                continue
            temp = np.argwhere(vocab_list == word)
            if temp.size == 1:
                word_indices.append(temp.min() + 1)
            # print(word)
    return word_indices


def emailFeatures(word_indices):
    '''将单词数字化'''
    indices = word_indices.copy()
    vocab = np.loadtxt('data/vocab.txt', dtype=str)
    vocab_list = vocab[:, 0].reshape(-1, 1)
    x = np.zeros(len(vocab_list))
    for i in range(len(indices)):
        indices[i] = indices[i] - 1
    x[indices] = 1
    return x


if __name__ == '__main__':
    # --------Part 1 Preprocessing Emails--------
    datas = load_data()
    word_indices = process_email(datas)
    # print(word_indices)
    print('Part 1', '*' * 50)

    # --------Part 2 Extracting Features from Emails(特征数字化)--------
    x = emailFeatures(word_indices)
    print('Part 2', '*' * 50)

    # --------Part 3 Training SVM for Spam Classification--------
    datas = loadmat('data/spamTrain.mat')
    X, y = datas['X'], datas['y']
    datas = loadmat('data/spamTest.mat')
    Xtest, ytest = datas['Xtest'], datas['ytest']

    model_linear = SVC(C=0.1, kernel='linear', degree=3)  # 使用高斯分类器可能会导致过拟合
    y = y.ravel()
    model_linear.fit(X, y)
    pred_y = model_linear.predict(X)
    acc = np.mean(pred_y == y, dtype=np.float)
    print('train acc = {:.3f}'.format(acc))
    # pred_test_y=model_linear.predict(Xtest).reshape(-1,1)
    # acc=np.mean(pred_test_y==ytest,dtype=np.float)
    # print('train acc = {:.3f}'.format(acc))
    print('Part 3', '*' * 50)

    # --------Part 4 Top Predictors for Spam--------
    # coef_中存在所有的词出现概率，argsort：返回按值从小达到的索引值
    index_array = np.argsort(model_linear.coef_).ravel()[::-1]  # b = a[i:j:s]表示：i,j与上面的一样，但s表示步进，缺省为1.
    vocab_list = np.loadtxt('data/vocab.txt', dtype=str)[:, 1]
    for i in range(15):
        print(vocab_list[index_array[i]], model_linear.coef_[:, index_array[i]])
    print('Part 4', '*' * 50)
