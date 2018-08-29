# -*- coding: utf-8 -*-

from fasttext import fasttext
from data.cnews_loader import *

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')


# def test():
#     print("start testing...")
#     # 处理测试数据
#     test_feature, test_target = process_file(test_dir, word_to_id, cat_to_id)
#     # test_feature, test_target = process_tfidf_file(test_dir, word_to_id, cat_to_id)  # 不能直接这样处理，应该取训练集的IDF值
#     test_predict = model.predict(test_feature)  # 返回预测类别
#     # test_predict_proba = model.predict_proba(test_feature)    # 返回属于各个类别的概率
#     # test_predict = np.argmax(test_predict_proba, 1)  # 返回概率最大的类别标签
#
#     # accuracy
#     true_false = (test_predict == test_target)
#     accuracy = np.count_nonzero(true_false) / float(len(test_target))
#     print()
#     print("accuracy is %f" % accuracy)
#
#     # precision    recall  f1-score
#     print()
#     print(metrics.classification_report(test_target, test_predict, target_names=categories))
#
#     # 混淆矩阵
#     print("Confusion Matrix...")
#     print(metrics.confusion_matrix(test_target, test_predict))


print("start training...")
fasttext_train_file = process_fasttext_file(train_dir, True)
model = fasttext.supervised(fasttext_train_file, "model/fasttext.model", epoch=50)

print("start testing...")
fasttext_test_file = process_fasttext_file(test_dir, False)
# model = fasttext.load_model('model/fasttext.model', label_prefix='__label__')
result = model.test(fasttext_test_file)
# model.predict_proba()
print(result.precision)
