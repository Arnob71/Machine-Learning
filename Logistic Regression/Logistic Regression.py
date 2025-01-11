import numpy as np
import copy
import matplotlib.pyplot as plt
def sigmoid(x):
    gz = 1/(1+np.exp(-x))
    return gz

def cost(x,y,w,b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        z = np.dot(w,x[i])+b
        cost += (-y[i]*np.log(sigmoid(z)))-((1-y[i])*np.log(1-sigmoid(z)))
    cost = cost/m
    return cost

def gradient(x,y,w,b):
    m,n = x.shape
    djdw = np.zeros((n,))
    djdb = 0.
    for i in range(m):
        err = sigmoid(np.dot(w,x[i])+b)-y[i]
        for j in range(n):
            djdw[j] += err*x[i][j]
        djdb += err
    djdw = djdw/m
    djdb = djdb/m
    return djdw, djdb

def gradientDescent(x,y,wIn,b,a,iter):
    w = copy.deepcopy(wIn)
    for i in range(iter):
        djdw, djdb = gradient(x,y,w,b)
        w -= a*djdw
        b -= a*djdb
    
    return w, b

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])
w_tmp  = np.zeros_like(X_train[0])
b = 1.
w, b = gradientDescent(X_train,y_train,w_tmp,b, 0.1, 10000)
print(w,b)
