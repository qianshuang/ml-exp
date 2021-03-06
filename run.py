# -*- coding: utf-8 -*-

from sklearn import metrics, svm, neural_network

from data.cnews_loader import *

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')


def train():
    print("start training...")
    # 处理训练数据，如果矩阵过大，可以采用Python scipy库中对稀疏矩阵的优化算法：scipy.sparse.csr_matrix((dd, (row, col)), )
    train_feature, train_target = process_file(train_dir, word_to_id, cat_to_id)  # 词频特征
    # train_feature, train_target = process_tfidf_file(train_dir, word_to_id, cat_to_id)  # TF-IDF特征
    # 模型训练
    model.fit(train_feature, train_target)


def test():
    print("start testing...")
    # 处理测试数据
    test_feature, test_target = process_file(test_dir, word_to_id, cat_to_id)
    # test_feature, test_target = process_tfidf_file(test_dir, word_to_id, cat_to_id)  # 不能直接这样处理，应该取训练集的IDF值
    test_predict = model.predict(test_feature)  # 返回预测类别
    # test_predict_proba = model.predict_proba(test_feature)    # 返回属于各个类别的概率
    # test_predict = np.argmax(test_predict_proba, 1)  # 返回概率最大的类别标签

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

categories, cat_to_id = read_category(test_dir)
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
# model = linear_model.LogisticRegression()   # ovr
# model = linear_model.LogisticRegression(multi_class="multinomial", solver="lbfgs")  # softmax回归
# SVM
# model = svm.LinearSVC()  # 线性，无概率结果
# model = svm.SVC()  # 核函数，训练慢
# MLP
model = neural_network.MLPClassifier(hidden_layer_sizes=(512, 128), max_iter=200, verbose=True, early_stopping=True)  # 注意max_iter是epoch数

train()
test()
