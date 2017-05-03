# -*- coding: utf-8 -*-
"""
Created on Mon May  1 07:46:01 2017

@author: sin-ev
"""

import random as rand
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA


def randomizing_centroid(data, numofK):
    # the whole dataset, numofk is to choose k inital locations
    index = rand.sample(range(len(data)), numofK)
    centroid = []
    for i in range(numofK):
        centroid.append(data[index[i]])
    return centroid
    '''
    centroid_cor = []
    index = rand.randint(0, 150)
    centroid_cor.append(data[index])
    index2 = -1
    while index2 == index:
        index2 = rand.randint(0, 150)
    centroid_cor.append(data[index2])
    index3 = -1
    while index3 == index or index3 == index2:
        index3 = rand.randint(0, 150)
    centroid_cor.append(data[index3])
    index4 = -1
    while index4 == index or index4 == index2 or index4 == index3:
        index4 = rand.randint(0, 150)
    centroid_cor.append(data[index4])
    return centroid_cor
'''


def eud(x1, x2):
    # calculate eud distance between x1 and x2
    dist = 0
    for (x, y) in zip(x1, x2):
        dist += (x - y) ** 2
    return dist ** 0.5


def euclidean_distance(centroids, data):
    # 1. find the distance between the data point and all for centroids
    # 2. find the point with minimum distance
    # 3. label the point with the minimum distance
    # 4. proceed this process for all data points
    # 5. return the labels????
    distances = []
    labels = []
    for i in range(len(data)):
        distance_data = []
        for j in range(len(centroids)):
            distance_data.append(eud(data[i], centroids[j]))
        distances.append(min(distance_data))
        labels.append(np.argmin(distance_data))
    return distances, labels


def update_centroids(labels, centroids, data):
    # 1. find mean for each label.
    # 2. replace the centroid with the mean
    #    print("before",centroids)
    for i in range(len(centroids)):
        centroids[i] = findmean(data, labels, i)
        #   print("after",centroids)
    return


def findmean(datapoints, labels, element):
    mean = [0 for x in range(4)]
    count = 0
    for i in range(150):
        if labels[i] == element:
            mean = np.add(mean, datapoints[i])
            count = count + 1
            #   print("printing mean of ", element)
            #   print(mean)
    if count != 0:
        for i in range(4):
            mean[i] = mean[i] / count
            #   print("after")
            #   print(mean)
    return mean


'''
    array = [0 for x in range(len(centroids))]
    for num in range(len(centroids)):
        for i in range(len(centroids)):
            array[num] = array[num] + (data[i] - centroids[num][i]) * (data[i] - centroids[num][i])
            #    print(array)
            #    print(np.argmin(array))
    return np.argmin(array)
'''


# def f1score(labels, truelabels, numofk):
#     true_pos, false_neg, false_pos = 0, 0, 0
#     recall, precision = 0, 0
#     score = [0] * 3
#     # k = [0 for x in range(numofk)]
#     for i in range(0, 49):
#         d = defaultdict(int)
#         for j in labels[0:50]:
#             d[j] += 1
#         result = max(d.iteritems(), key=lambda x: x[1])
#         maximum = result[0]
#         true_pos = result[1]
#         false_neg = 50 - true_pos
#         false_pos = labels.count(maximum) - true_pos
#         # maximum = np.argmax(labels[0:50])
#         # maximum = labels[maximum_idx]
#         # true_pos = labels[0:50].count(maximum)
#         # false_neg = 50 - true_pos
#         # false_pos = labels[51:150].count(maximum)
#         recall = float(true_pos) / float(true_pos + false_neg)
#         precision = float(true_pos) / float(true_pos + false_pos)
#     score[0] = 2 * precision * recall / (precision + recall)
#
#     for i in range(50, 99):
#         d = defaultdict(int)
#         for j in labels[51:100]:
#             d[j] += 1
#         result = max(d.iteritems(), key=lambda x: x[1])
#         maximum = result[0]
#         true_pos = result[1]
#         false_neg = 50 - true_pos
#         false_pos = labels.count(maximum) - true_pos
#         recall = float(true_pos) / float(true_pos + false_neg)
#         precision = float(true_pos) / float(true_pos + false_pos)
#     score[1] = 2 * precision * recall / (precision + recall)
#
#     for i in range(100, 149):
#         d = defaultdict(int)
#         for j in labels[101:150]:
#             d[j] += 1
#         result = max(d.iteritems(), key=lambda x: x[1])
#         maximum = result[0]
#         true_pos = result[1]
#         false_neg = 50 - true_pos
#         false_pos = labels.count(maximum) - true_pos
#         recall = float(true_pos) / float(true_pos + false_neg)
#         precision = float(true_pos) / float(true_pos + false_pos)
#     score[2] = 2 * precision * recall / (precision + recall)
#
#     print("F1 score: ", score)
#     return score


def parta(numofK, repeat, data):
    numIt = [5, 10, 20]
    # numofK = 3
    # repeat = 4
    centroids = randomizing_centroid(data, numofK)

    # create an array of randomized centroids

    # centroids = randomizing_centroid(data, numofK)
    # print ("randomized centroid")
    # print (centroids)

    # find out which centroid is closest for each data points

    bestc = []
    bestloss = []
    bestlabels = []
    for i in range(len(numIt)):
        all_centroids = []
        for j in range(repeat):
            centroids = randomizing_centroid(data, numofK)
            for k in range(numIt[i]):
                [distances, labels] = euclidean_distance(centroids, data)
                update_centroids(labels, centroids, data)
                all_centroids.append(centroids)
                # select the best
        loss = []
        labelss = []
        for q in range(len(all_centroids)):
            [distances, labels] = euclidean_distance(all_centroids[q], data)
            loss.append(sum(distances))
            labelss.append(labels)
        bestc.append(all_centroids[np.argmin(loss)])
        bestloss.append(min(loss))
        bestlabels.append(labelss[np.argmin(loss)])
    return bestc, bestloss, bestlabels


def lossPlot(bestloss, numofK, repeat):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
        xs = np.arange(20)
        # ys = np.random.rand(20)
        ys = bestloss

        # You can provide either a single color or an array. To demonstrate this,
        # the first bar of each set will be colored cyan.
        cs = [c] * len(xs)
        cs[0] = 'c'
        ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

    ax.set_xlabel('K')
    ax.set_ylabel('Iteration')
    ax.set_zlabel('SS_Total')
    plt.show()


# main
def main_helper(numofk, repeat):
    print("Calculating K = %d, repeat = %d ..." % (numofk, repeat))
    numVector, numDimension = 150, 4
    truelabels = ["a" for x in range(numVector)]
    labels = [0 for x in range(numVector)]
    element = 0
    data = [[0 for x in range(numDimension)] for y in range(numVector)]
    # get the data into array of [150][4]
    with open('iris.txt') as f:
        for line in f:
            lines = line.split(",")
            data[element][0] = (float(lines[0]))
            data[element][1] = (float(lines[1]))
            data[element][2] = (float(lines[2]))
            data[element][3] = (float(lines[3]))
            labels[element] = lines[4]
            element = element + 1
    bestc, bestloss, bestlabels = parta(numofk, repeat, data)
    return bestc, bestloss, bestlabels


# plot ss_total in part a
# repeat = 4, 10, 20
# k = 3, 4, 8
def main():
    k_array = [3, 4, 8]
    repeat_array = [4, 10, 20]
    for repeat_num in range(len(repeat_array)):
        bestloss_array = []
        for i in range(len(k_array)):
            bestc, bestloss, bestlabels = main_helper(k_array[i], repeat_array[repeat_num])
            bestloss_array.append(bestloss)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        dx = np.ones(10) * 0.4
        dy = np.ones(10) * 1
        # zpos = [0, 0, 0]
        ys = []
        for i in range(len(bestloss_array)):
            ys.append(np.asarray(bestloss_array[i]))
        xs = []
        for i in range(len(k_array)):
            xs.append([k_array[i]] * 3)
        c = ['r', 'g', 'b']
        ax.set_xlabel('K')
        ax.set_ylabel('Iteration')
        ax.set_zlabel('SS_Total')
        plt.title('Stop condition: repeat %d times' % repeat_array[repeat_num])
        for i in range(len(xs)):
            ax.bar3d(xs[i], [5, 10, 20], [0, 0, 0], dx, dy, ys[i], color=c[i])
    plt.show()
    print("Finished part 1 plotting")


if __name__ == "__main__":
    '''
    scores = []
    for i in range(3):
        score = f1score(bestlabels[i], truelabels,numofk)
        scores.append(score)'''
    main()
