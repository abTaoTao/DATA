import text2vec
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

iris_df = datasets.load_iris()
model = KMeans(n_clusters=3)
model.fit(iris_df.data)
pre = model.predict(iris_df.data)
x_axis = iris_df.data[:, 0]
y_axis = iris_df.data[:, 2]


