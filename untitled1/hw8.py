# -*- coding: utf-8 -*-
"""
Created on Mon May  1 07:46:01 2017

@author: sin-ev
"""

import random as rand
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def randomizing_centroid(data, numofK):
#the whole dataset, numofk is to choose k inital locations
    index = rand.sample(range(len(data)),numofK)
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
def eud(x1,x2):
    #calculate eud distance between x1 and x2
    dist = 0
    for (x,y) in zip(x1,x2):
        dist += (x-y)**2
    return dist**0.5

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
            distance_data.append(eud(data[i],centroids[j]))
        distances.append(min(distance_data))
        labels.append(np.argmin(distance_data))
    return distances,labels
            
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
def returnindex(labels):
    index0 = []
    index1 = []
    index2 = []
    index = []
    for i in range(len(labels)):
        if labels[i] == 0:
            index0.append(i)
        elif labels[i] == 1:
            index1.append(i)
        elif labels[i] == 2:
            index2.append(i)
    index.append(index0)
    index.append(index1)
    index.append(index2)
    return index

def fscore(labels,truelabels):
    il = returnindex(labels)
    ilmean = []
    it = returnindex(truelabels)
    itmean = []
    for i in range(3):
        ilmean.append(np.mean(il[i]))
        itmean.append(np.mean(it[i]))
    indexil = np.argsort(ilmean)
    indexit = np.argsort(itmean)
    newil = []
    newit = []
    for i in range(3):
        newil.append(il[indexil[i]])
        newit.append(it[indexit[i]])
    score = []
    for i in range(3):
        tempscore = 0
        test = list(set(newil[i]).intersection(set(newit[i])))
        Recall = len(test)/len(newit[i])
        Precision = len(test)/len(newil[i])
        tempscore = 2	*	Precision	*	Recall	/	(Precision	+	Recall)
        score.append(tempscore)
    return newil,newit,score
'''    
def f1score(labels, truelabels,numofk):
    true_pos, false_neg, false_pos = 0, 0, 0
    recall, precision = 0, 0
    score = [0] * 3
    # k = [0 for x in range(numofk)]
    for i in range(0,49):
        d = defaultdict(int)
        for j in labels[0:50]:
            d[j] += 1
        result = max(d.iteritems(), key=lambda x: x[1])
        maximum = result[0]
        true_pos = result[1]
        false_neg = 50 - true_pos
        false_pos = labels.count(maximum) - true_pos
        # maximum = np.argmax(labels[0:50])
        # maximum = labels[maximum_idx]
        # true_pos = labels[0:50].count(maximum)
        # false_neg = 50 - true_pos
        # false_pos = labels[51:150].count(maximum)
        recall = float(true_pos) / float(true_pos + false_neg)
        precision = float(true_pos) / float(true_pos + false_pos)
    score[0] = 2 * precision * recall / (precision + recall)

    for i in range(50,99):
        d = defaultdict(int)
        for j in labels[51:100]:
            d[j] += 1
        result = max(d.iteritems(), key=lambda x: x[1])
        maximum = result[0]
        true_pos = result[1]
        false_neg = 50 - true_pos
        false_pos = labels.count(maximum) - true_pos
        recall = float(true_pos) / float(true_pos + false_neg)
        precision = float(true_pos) / float(true_pos + false_pos)
    score[1] = 2 * precision * recall / (precision + recall)

    for i in range(100,149):
        d = defaultdict(int)
        for j in labels[101:150]:
            d[j] += 1
        result = max(d.iteritems(), key=lambda x: x[1])
        maximum = result[0]
        true_pos = result[1]
        false_neg = 50 - true_pos
        false_pos = labels.count(maximum) - true_pos
        recall = float(true_pos) / float(true_pos + false_neg)
        precision = float(true_pos) / float(true_pos + false_pos)
    score[2] = 2 * precision * recall / (precision + recall)

    print("F1 score: ", score)
    return score
'''

def parta(numofK,data):
        
    numIt = [5,10,20]        
    #numofK = 3
    repeat = 4
    centroids = randomizing_centroid(data, numofK)

    #create an array of randomized centroids 

    #centroids = randomizing_centroid(data, numofK)
    #print ("randomized centroid")
    #print (centroids)   
        
    #find out which centroid is closest for each data points

    bestc = []
    bestloss = []
    bestlabels = []
    for i in range(len(numIt)):
        all_centroids = []
        for j in range(repeat):
            centroids = randomizing_centroid(data, numofK)
            for k in range(numIt[i]):
                [distances,labels] = euclidean_distance(centroids, data)
                update_centroids(labels, centroids, data)
                all_centroids.append(centroids)
    #select the best
        loss = []
        labelss = []
        for q in range(len(all_centroids)):
            [distances,labels] = euclidean_distance(all_centroids[q], data)
            loss.append(sum(distances))
            labelss.append(labels)
        bestc.append(all_centroids[np.argmin(loss)])
        bestloss.append(min(loss))
        bestlabels.append(labelss[np.argmin(loss)])
    return bestc,bestloss,bestlabels

def finde(data,index):
    newdata = []
    for i in range(len(index)):
        newdata.append(data[index[i]])
    return newdata

#main
def main(numofk,data):
    scores = []
    for i in range(len(bestlabels)):        
        newil,newit,score = fscore(bestlabels[i],truelabels)
        scores.append(score)
    return bestc,bestloss,bestlabels,scores
'''
scores = []
for i in range(3):
    score = f1score(bestlabels[i], truelabels,numofk)
    scores.append(score)'''
numVector, numDimension = 150, 4
truelabels = [0 for x in range(numVector)]
labels = [0 for x in range(numVector)] 
element = 0
data = [[0 for x in range(numDimension)] for y in range(numVector)]
    #get the data into array of [150][4]
with open('iris.txt') as f:
    for line in f:
        lines = line.split(",")
        data[element][0] = (float(lines[0]))
        data[element][1] = (float(lines[1]))
        data[element][2] = (float(lines[2]))
        data[element][3] = (float(lines[3]))
        labels[element]  = lines[4]
        element = element + 1
setlabels = set(labels)
setlabels = list(setlabels)
#make labels to be numbers
for i in range(len(labels)):
    tempindex = setlabels.index(labels[i])
    truelabels[i] = tempindex

numofk = 3
bestc,bestloss,bestlabels = parta(numofk,data)
#bestc,bestloss,bestlabels,score = main(numofk,data)
if numofk == 3:
    scores = []
    for i in range(len(bestlabels)):        
        newil,newit,score = fscore(bestlabels[i],truelabels)
        scores.append(score)
    meanscores = np.mean(scores,1)
    maxscore_index = np.argmax(meanscores)
    best_centroid = bestc[maxscore_index]
    best_labels = bestlabels[maxscore_index]
    x0 = []
    x1 = []
    x2 = []
    x3 = []
    for i in range(len(data)):
        x0.append(data[i][0])
        x1.append(data[i][1])
        x2.append(data[i][2])
        x3.append(data[i][3])
    for i in range(3):
        #a. Sepal	Length	vs	Sepal	Width	vs	Petal	Length
        ax=plt.subplot(111,projection='3d')
        #123
        x = x0
        y = x1
        z = x2
        colors = ['y','r','g']
        for i in range(3):
            ax.scatter(finde(x,newil[i]),finde(y,newil[i]),finde(z,newil[i]),c=colors[i]) #绘制数据点
            ax.scatter(best_centroid[i][0],best_centroid[i][1],best_centroid[i][2],c=colors[i],marker='^')   
        #b. Sepal	Length	vs	Sepal	Width	vs	Petal	Width
        ax=plt.subplot(111,projection='3d')
        #124
        x = x0
        y = x1
        z = x3
        colors = ['y','r','g']
        colorss = ['black','violet','blue']
        for i in range(3):
            ax.scatter(finde(x,newil[i]),finde(y,newil[i]),finde(z,newil[i]),c=colors[i]) #绘制数据点
            ax.scatter(best_centroid[i][0],best_centroid[i][1],best_centroid[i][3],c=colorss[i],marker='^')   

        #c. Sepal	Length	vs	Petal	Length	vs	Petal	Width
        ax=plt.subplot(111,projection='3d')
        #134
        x = x0
        y = x2
        z = x3
        for i in range(3):
            ax.scatter(finde(x,newil[i]),finde(y,newil[i]),finde(z,newil[i]),c=colors[i]) #绘制数据点
            ax.scatter(best_centroid[i][0],best_centroid[i][2],best_centroid[i][3],c=colorss[i],marker='^')   

        #d. Sepal	Width	vs	Petal Length	vs	Petal	Width
        ax=plt.subplot(111,projection='3d')
        #234
        x = x1
        y = x2
        z = x3
        for i in range(3):
            ax.scatter(finde(x,newil[i]),finde(y,newil[i]),finde(z,newil[i]),c=colors[i]) #绘制数据点
            ax.scatter(best_centroid[i][1],best_centroid[i][2],best_centroid[i][3],c=colorss[i],marker='^')   

        