#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import random

import numpy as np
from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GMM

from kmeans import kmeans

iris = datasets.load_iris()
dataset = iris.data
target = iris.target


def cluster_accuracy(y_true, y_pred):
	correct = 0.0
	classes = np.unique(y_true)
	for c in classes:
		idx_true = np.where(y_true == c)
		cluster = y_pred[idx_true]
		dominant = stats.mode(cluster)[0][0]
		correct_idxs = np.where((cluster == dominant) == True)[0]
		correct += len(correct_idxs)
	return correct/len(y_true)
	


centres, y_pred, distances = kmeans(dataset, np.array(random.sample(list(dataset), 3)), metric="euclidean")
accuracy = cluster_accuracy(target, y_pred)
print("Iris: K-Means (Distância Euclideana): Acurácia: %0.3f" % (accuracy))

centres, y_pred, distances = kmeans(dataset, np.array(random.sample(list(dataset), 3)), metric="cityblock")
accuracy = cluster_accuracy(target, y_pred)
print("Iris: K-Means (Distância de Manhattan): Acurácia: %0.3f" % (accuracy))

ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')
y_pred = ac.fit_predict(dataset)
accuracy = cluster_accuracy(target, y_pred)
print("Iris: Hierárquico (Distância Euclideana): Acurácia: %0.3f" % (accuracy))

ac = AgglomerativeClustering(n_clusters=3, affinity='manhattan', linkage='average')
y_pred = ac.fit_predict(dataset)
accuracy = cluster_accuracy(target, y_pred)
print("Iris: Hierárquico (Distância de Manhattan): Acurácia: %0.3f" % (accuracy))


dataset = np.genfromtxt('spiral.txt', delimiter="\t", dtype=np.float64)
target = dataset[:,-1]
dataset = dataset[:,0:-1]

centres, y_pred, distances = kmeans(dataset, np.array(random.sample(list(dataset), 3)), metric="euclidean")
accuracy = cluster_accuracy(target, y_pred)
print("Spiral: K-Means (Distância Euclideana): Acurácia: %0.3f" % (accuracy))

plt.figure()
plt.clf()
classes = [0, 1, 2] 
for c, i, classe in zip("rgb", classes, classes):
    plt.scatter(dataset[y_pred == i, 0], dataset[y_pred == i, 1], c=c, label=classe)
plt.legend()
plt.title('Spiral (K-Means) (Dist. Euclideana)')

ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean')
y_pred = ac.fit_predict(dataset)
accuracy = cluster_accuracy(target, y_pred)
print("Spiral: Hierárquico (Distância Euclideana): Acurácia: %0.3f" % (accuracy))

plt.figure()
plt.clf()
classes = [0, 1, 2] 
for c, i, classe in zip("rgb", classes, classes):
    plt.scatter(dataset[y_pred == i, 0], dataset[y_pred == i, 1], c=c, label=classe)
plt.legend()
plt.title('Spiral (Hierárquico) (Dist. Euclideana)')



dataset = np.genfromtxt('jain.txt', delimiter="\t", dtype=np.float64)
target = dataset[:,-1]
dataset = dataset[:,0:-1]

centres, y_pred, distances = kmeans(dataset, np.array(random.sample(list(dataset), 2)), metric="euclidean")
accuracy = cluster_accuracy(target, y_pred)
print("Jain: K-Means (Distância Euclideana): Acurácia: %0.3f" % (accuracy))

plt.figure()
plt.clf()
classes = [0, 1] 
for c, i, classe in zip("rg", classes, classes):
    plt.scatter(dataset[y_pred == i, 0], dataset[y_pred == i, 1], c=c, label=classe)
plt.legend()
plt.title('Jain K-Means (Dist. Euclideana)')

ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean')
y_pred = ac.fit_predict(dataset)
accuracy = cluster_accuracy(target, y_pred)
print("Jain: Hierárquico (Distância Euclideana): Acurácia: %0.3f" % (accuracy))

plt.figure()
plt.clf()
classes = [0, 1] 
for c, i, classe in zip("rg", classes, classes):
    plt.scatter(dataset[y_pred == i, 0], dataset[y_pred == i, 1], c=c, label=classe)
plt.legend()
plt.title('Jain (Hierárquico) (Dist. Euclideana)')

dataset = np.genfromtxt('aggregation.txt', delimiter="\t", dtype=np.float64)
target = dataset[:,-1]
dataset = dataset[:,0:-1]

def make_ellipses(gmm, ax, colors):
    for n, color in colors:
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        #v *= 9
        print("ELIPSE")
        print(gmm.means_[n, :2])
        print(v[0], v[1])
        print(180 + angle)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.4)
        ax.add_artist(ell)


classifiers = [4, 7, 10]
n_classifiers = len(classifiers)
for idx, n_components in enumerate(classifiers):
	fig = plt.figure()
	gmm = GMM(n_components=n_components, init_params='wc', n_iter=20, n_init=20)
	gmm.fit(dataset)
	
	
	ax = fig.add_subplot(1,1,1, aspect='equal')
	print(n_components)
	colors = 'rgbcmyk'[0:n_components]
	make_ellipses(gmm, ax, enumerate(colors))
	
	y_pred = gmm.predict(dataset)
	accuracy = cluster_accuracy(target, y_pred)
	for n, color in enumerate(colors):
		data = dataset[y_pred == n]
		plt.plot(data[:, 0], data[:, 1], 'x', color=color)
		
	plt.xticks(())
	plt.yticks(())
	plt.title("%d Componentes" % (n_components))

plt.legend()
plt.show()
