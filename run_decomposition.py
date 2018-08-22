# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation

# 假设我们有4个词，5个文本组成的特征值矩阵
X = np.array([[1, 1, 5, 2, 3], [0, 6, 2, 1, 1], [3, 4, 0, 3, 1], [4, 1, 5, 6, 3]])
# n_components即我们的主题数k，确定它需要一些对于要分析文本主题大概的先验知识
# model = NMF(n_components=2, alpha=0.01)
model = LatentDirichletAllocation(n_components=2, random_state=0, learning_method='batch')

# 单词话题矩阵
W = model.fit_transform(X)
# 文本话题矩阵
H = model.components_
# 模型的困惑度，代表生成整个训练集文档的似然估计的负值，perplexity值越小，模型能力越强。可以用来作为选择超参K的参考。
perplexity = model.perplexity(X)

print(W)
print(H)
print(perplexity)
