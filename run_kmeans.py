# -*- coding: utf-8 -*-
from sklearn import metrics
from sklearn.cluster import KMeans

from data.cnews_loader import *

base_dir = 'data/cnews'
test_dir = os.path.join(base_dir, 'cnews.kmeans.test.txt')
vocab_dir = os.path.join(base_dir, 'cnews.kmeans.vocab.txt')


if not os.path.exists(vocab_dir):
    # 构建词典表
    build_vocab(test_dir, vocab_dir)

categories, cat_to_id = read_category(test_dir)
words, word_to_id = read_vocab(vocab_dir)


print("start doing k-means...")
# 处理数据
feature, target = process_file(test_dir, word_to_id, cat_to_id)
# 训练k-means聚类
kmeans = KMeans(n_clusters=10, random_state=0).fit(feature)  # random_state为随机数种子，若不设置每次运行结果不一样
print(kmeans.predict([feature[0]]))  # 预测簇id
print(kmeans.cluster_centers_)  # 聚类中心
print(kmeans.labels_)  # 返回所有簇id
print(metrics.calinski_harabaz_score(feature, kmeans.predict(feature)))  # Calinski-Harabasz分数可以用来评估聚类效果，它内部使用簇内的稠密程度和簇间的离散程度的比值，所以数值越大效果也好
