# -*- coding: utf-8 -*-

from sklearn import neighbors
from sklearn import metrics
from data.cnews_loader import *

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')


def train():
    # 处理训练数据
    train_feature, train_target = process_file(train_dir, word_to_id, cat_to_id)
    model.fit(train_feature, train_target)


def test():
    test_feature, test_target = process_file(test_dir, word_to_id, cat_to_id)
    test_predict = model.predict(test_feature)

    # accuracy
    true_false = (test_predict == test_target)
    accuracy = np.count_nonzero(true_false) / float(len(test_target))
    print()
    print("accuracy is %f" % accuracy)

    # precision    recall  f1-score
    print()
    print(metrics.classification_report(test_target, test_predict, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    print(metrics.confusion_matrix(test_target, test_predict))


if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
    build_vocab(train_dir, vocab_dir)

categories, cat_to_id = read_category()
words, word_to_id = read_vocab(vocab_dir)

# kNN
model = neighbors.KNeighborsClassifier()

train()
test()
