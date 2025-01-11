import numpy as np
import matplotlib.pyplot as plt

def findClosestCentroids(x,centroids):
    m = x.shape[0]
    n = centroids.shape[0]
    idx = np.zeros(x.shape[0],dtype=int)
    for i in range(m):
        distance =[]
        for j in range(n):
            norm_ij = np.linalg.norm(x[i]-centroids[j])
            distance.append(norm_ij)
        idx[i] = np.argmin(distance)
    return idx

def computeCentroids(x, idx, k):
    m, n = x.shape

    centroids = np.zeros((k,n))

    for i in range(k):
        points = x[idx==i]
        centroids[i] = np.mean(points, axis=0)

    return centroids
def initCentroids(x,k):
    randIdx = np.random.permutation(x.shape[0])
    centroids = x[randIdx[:k]]

    return centroids

def kMeans(x, initialCentroids, iterations):
    k = initialCentroids.shape[0]
    centroids = initialCentroids
    idx= np.zeros(x.shape[0])
    for i in range(iterations):
        idx = findClosestCentroids(x,centroids)
        centroids = computeCentroids(x,idx,k)
    return centroids, idx

img = plt.imread("bird_small.png")
img = img/255
reshape = np.reshape(img,(img.shape[0]*img.shape[1],3))
centroids, idx = kMeans(reshape, initCentroids(reshape,128), 10)
after = centroids[idx,:]
after = np.reshape(after,img.shape)
plt.imshow(after*255)
plt.show()