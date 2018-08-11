# -*- coding: utf-8 -*-

from sklearn import neighbors, tree, ensemble, naive_bayes, linear_model
import xgboost
from sklearn import metrics
from data.cnews_loader import *

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')


def train():
    print("start training...")
    # 处理训练数据
    # train_feature, train_target = process_file(train_dir, word_to_id, cat_to_id)  # 词频特征
    train_feature, train_target = process_tfidf_file(train_dir, word_to_id, cat_to_id)  # TF-IDF特征
    # 模型训练
    model.fit(train_feature, train_target)


def test():
    print("start testing...")
    # 处理测试数据
    test_feature, test_target = process_file(test_dir, word_to_id, cat_to_id)
    # test_predict = model.predict(test_feature)  # 返回预测类别
    test_predict_proba = model.predict_proba(test_feature)    # 返回属于各个类别的概率
    test_predict = np.argmax(test_predict_proba, 1)  # 返回概率最大的类别标签

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


if not os.path.exists(vocab_dir):
    # 构建词典表
    build_vocab(train_dir, vocab_dir)

categories, cat_to_id = read_category()
words, word_to_id = read_vocab(vocab_dir)

# kNN
# model = neighbors.KNeighborsClassifier()
# decision tree
# model = tree.DecisionTreeClassifier()
# random forest
# model = ensemble.RandomForestClassifier(n_estimators=10)  # n_estimators为基决策树的数量，一般越大效果越好直至趋于收敛
# AdaBoost
# model = ensemble.AdaBoostClassifier(learning_rate=1.0)  # learning_rate的作用是收缩基学习器的权重贡献值
# GBDT
# model = ensemble.GradientBoostingClassifier(n_estimators=10)
# xgboost
# model = xgboost.XGBClassifier(n_estimators=10)
# Naive Bayes
# model = naive_bayes.MultinomialNB()
# logistic regression
model = linear_model.LogisticRegression()   # ovr
model = linear_model.LogisticRegression(multi_class="multinomial", solver="lbfgs")  # softmax回归

train()
test()
