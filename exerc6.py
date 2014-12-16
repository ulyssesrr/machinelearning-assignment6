#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import random

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

from sklearn import datasets

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
	


centres, y_pred, distances = kmeans(dataset, np.array(random.sample(dataset, 3)), metric="euclidean")
accuracy = cluster_accuracy(target, y_pred)
print("Iris: K-Means (Distância Euclideana): Acurácia: %0.3f" % (accuracy))

centres, y_pred, distances = kmeans(dataset, np.array(random.sample(dataset, 3)), metric="cityblock")
accuracy = cluster_accuracy(target, y_pred)
print("Iris: K-Means (Distância de Manhattan): Acurácia: %0.3f" % (accuracy))

print("TODO: Iris: Hierárquico (Distância Euclideana): Acurácia: %0.3f" % (accuracy))
print("TODO: Iris: Hierárquico (Distância de Manhattan): Acurácia: %0.3f" % (accuracy))


dataset = np.genfromtxt('spiral.txt', delimiter="\t", dtype=np.float64)
target = dataset[:,-1]
dataset = dataset[:,0:-1]

centres, y_pred, distances = kmeans(dataset, np.array(random.sample(dataset, 3)), metric="euclidean")
accuracy = cluster_accuracy(target, y_pred)
print("Spiral: K-Means (Distância Euclideana): Acurácia: %0.3f" % (accuracy))
print("TODO: Spiral: Hierárquico (Distância Euclideana): Acurácia: %0.3f" % (accuracy))

plt.figure()
plt.clf()
classes = [0, 1, 2] 
for c, i, classe in zip("rgb", classes, classes):
    plt.scatter(dataset[y_pred == i, 0], dataset[y_pred == i, 1], c=c, label=classe)
plt.legend()
plt.title('Spiral K-Means (Dist. Euclideana)')
#plt.show()


dataset = np.genfromtxt('jain.txt', delimiter="\t", dtype=np.float64)
target = dataset[:,-1]
dataset = dataset[:,0:-1]

centres, y_pred, distances = kmeans(dataset, np.array(random.sample(dataset, 2)), metric="euclidean")
accuracy = cluster_accuracy(target, y_pred)
print("Jain: K-Means (Distância Euclideana): Acurácia: %0.3f" % (accuracy))
print("TODO: Jain: Hierárquico (Distância Euclideana): Acurácia: %0.3f" % (accuracy))

plt.figure()
plt.clf()
classes = [0, 1] 
for c, i, classe in zip("rg", classes, classes):
    plt.scatter(dataset[y_pred == i, 0], dataset[y_pred == i, 1], c=c, label=classe)
plt.legend()
plt.title('Jain K-Means (Dist. Euclideana)')
plt.show()
