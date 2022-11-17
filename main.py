import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from datetime import datetime
import time



data = pd.read_csv("insurance.csv", delimiter=",")

data.drop("region", axis=1, inplace=True)
data.drop("children", axis=1, inplace=True)
#data.drop("smoker", axis=1, inplace=True)
data.sex.replace({'male': 1, 'female': 0}, inplace=True)
data.smoker.replace({'yes': 1, 'no': 0}, inplace=True)
data.charges = data.charges / 1000

print("\n", data.describe())

models = []
score1 = []
score2 = []
for i in range(2, 10):
    model = KMeans(n_clusters=i, random_state=123, init='k-means++').fit(data)
    models.append(model)
    score1.append(model.inertia_)
    score2.append(silhouette_score(data, model.labels_))

plt.grid()
plt.plot(np.arange(2, 10), score1, marker='o')
plt.show()

plt.grid()
plt.plot(np.arange(2, 10), score2, marker='o')
plt.show()

#KMeans

kmeans_time_start = datetime.now()
model1 = KMeans(n_clusters=3, random_state=123, init='k-means++')
model1.fit(data)
print(model1.cluster_centers_)

labels = model1.labels_
data['Cluster'] = labels


print("\n", data['Cluster'].value_counts())

fig = go.Figure(data=[
    go.Scatter3d(x=data['age'], y=data['bmi'], z=data['charges'], mode='markers', marker_color=data['Cluster'],
                 marker_size=4)])
fig.show()
kmeans_time_end = datetime.now()
# Агломеративаня кластеризация

agg_time_start = datetime.now()
model2 = AgglomerativeClustering(3, compute_distances=True)
clastering = model2.fit(data)
data['Cluster'] = clastering.labels_

fig = go.Figure(data=[
    go.Scatter3d(x=data['age'], y=data['bmi'], z=data['charges'], mode='markers', marker_color=data['Cluster'],
                 marker_size=4)])
fig.show()
agg_time_end = datetime.now()
# DBSCAN

dbscan_time_start = datetime.now()
model3 = DBSCAN(eps=11, min_samples=5).fit(data)
data['Cluster'] = model3.labels_

fig = go.Figure(data=[
    go.Scatter3d(x=data['age'], y=data['bmi'], z=data['charges'], mode='markers', marker_color=data['Cluster'],
                 marker_size=4)])
fig.show()
dbscan_time_end = datetime.now()

print("Kmean time: ", kmeans_time_end-kmeans_time_start)
print("Agglomerative time: ", agg_time_end-agg_time_start)
print("DBSCAN time: ", dbscan_time_end-dbscan_time_start)