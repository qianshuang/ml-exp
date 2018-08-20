# -*- coding: utf-8 -*-
import re
from collections import Counter
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfTransformer

base_dir = 'data'
stopwords_dir = os.path.join(base_dir, 'stop_words.txt')


def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')


def stopwords_list(filename):
    stopwords = []
    with open_file(filename) as f:
        for line in f:
            try:
                content = line.strip()
                stopwords.append(content)
            except:
                pass
    return stopwords


stopwords = stopwords_list(stopwords_dir)


def remove_stopwords(content):
    return list(set(content).difference(set(stopwords)))


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                contents.append(content)
                labels.append(label)
            except:
                pass
    return contents, labels


def build_vocab(total_dir, vocab_dir):
    """根据训练集构建词汇表，存储"""
    print("building vacab...")
    words = []
    with open_file(total_dir) as f:
        for line in f:
            sents = list(line.strip().split('\t')[1])
            for sent in sents:
                # words.extend(number_norm(sent))
                words.extend(sent)
    words = remove_stopwords(words)
    # counter = Counter(words)
    # count_pairs = counter.most_common(5000)
    # words, _ = list(zip(*count_pairs))
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def number_norm(x):
    re_num = re.compile(r'\d')
    if re_num.match(x):
        return 'NUM'


def read_vocab(vocab_dir):
    """读取词汇表"""
    print("read_vocab...")
    words = open_file(vocab_dir).read().strip().split('\n')
    word_to_id = dict(zip(words, range(0, len(words))))

    return words, word_to_id


def read_category(cat_dir):
    """读取所有类别"""
    print("read_category...")
    # categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_set = set()
    with open_file(cat_dir) as f:
        for line in f:
            cat_set.add(line.split("\t")[0].strip())
    categories = list(cat_set)
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    Arguments:
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    Returns:
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def process_file(filename, word_to_id, cat_to_id):
    contents, labels = read_file(filename)

    vocab_size = len(word_to_id)
    data = np.zeros((len(contents), vocab_size)).tolist()
    label = []
    for i in range(len(contents)):
        words = list(contents[i].strip())
        words = remove_stopwords(words)
        dd = [word_to_id[x] for x in words if x in word_to_id]
        counter = Counter(dd)
        for k, v in counter.items():
            data[i][k] = v
        label.append(cat_to_id[labels[i]])
    return np.array(data), np.array(label)


def word2features(words, i):
    # TODO 还可以取词的ngram特征

    word = words[i]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),  # 当前词
        # 'word[-3:]': word[-3:],
        # 'word[-2:]': word[-2:],
        # 'word.isupper()': word.isupper(),
        # 'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        # 'postag': postag,
        # 'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = words[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),  # 当前词的前一个词
            # '-1:word.istitle()': word1.istitle(),
            # '-1:word.isupper()': word1.isupper(),
            # '-1:postag': postag1,
            # '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(words)-1:
        word1 = words[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),  # 当前词的后一个词
            # '+1:word.istitle()': word1.istitle(),
            # '+1:word.isupper()': word1.isupper(),
            # '+1:postag': postag1,
            # '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def process_crf_file(crf_train_source_dir, crf_train_target_dir):
    features = []
    labels = []
    with open_file(crf_train_source_dir) as f:
        for line in f:
            feature = []
            words = re.split('\s+', line.strip())
            for i in range(len(words)):
                feature.append(word2features(words, i))
            features.append(feature)

    with open_file(crf_train_target_dir) as f:
        for line in f:
            label = []
            ls = re.split('\s+', line.strip())
            for i in range(len(ls)):
                label.append(ls[i])
            labels.append(label)

    # for i in range(len(features)):
    #     if len(features[i]) != len(labels[i]):
    #         print("The numbers of items and labels differ, line " + str(i))

    return np.array(features), np.array(labels)


def process_tfidf_file(filename, word_to_id, cat_to_id):
    data, label = process_file(filename, word_to_id, cat_to_id)
    data_tfidf = TfidfTransformer().fit_transform(data)
    return data_tfidf, label


def process_maxent_file(filename, word_to_id, cat_to_id):
    data = []
    contents, labels = read_file(filename)

    for i in range(len(contents)):
        words = list(contents[i].strip())
        words = remove_stopwords(words)
        dd = [word_to_id[x] for x in words if x in word_to_id]
        counter = Counter(dd)
        wordid_freq = {}
        for k, v in counter.items():
            wordid_freq[k] = v / float(len(dd))

        data.append((wordid_freq, cat_to_id[labels[i]]))
    return data
