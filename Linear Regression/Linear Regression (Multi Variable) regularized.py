import numpy as np
import math, copy

def zScroreNorm(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    xNorm = (x-mu)/sigma
    return xNorm
def predict(x,w,b):
    y = np.dot(w,x)+b
    return y

def costFunction(x,y,w,b):
    m = x.shape[0]
    sum = 0
    for i in range(m):
        fwb = np.dot(x[i],w)+b
        sum += (fwb-y[i])**2
    cost = sum/(2*m)
    return cost

def costFunctionWithReg(x,y,w,b,lmd):
    m,n = x.shape
    sum = 0
    for i in range(m):
        fwb = np.dot(x[i],w)+b
        sum += (fwb-y[i])**2
    cost = sum/(2*m)

    sumWjSqr = 0
    for j in range(n):
        sumWjSqr += w[j]**2
    regCost = (lmd/(2*m))*sumWjSqr
    return cost+regCost

def gradient(x,y,w,b,lmd):
    m,n = x.shape
    djdw = np.zeros((n))
    djdb = 0
    for i in range (m):
        fwb = np.dot(x[i],w)+b-y[i]
        for j in range(n):
            djdw[j] += fwb*x[i][j]
        djdb += fwb
    djdw = djdw/m
    reg = (lmd/m)*w
    djdb = djdb/m

    return djdw+reg, djdb
def gradientDescent(numOfIter,x,y,wIn,bIn,a,lmd):
    history = []
    cost = []
    w = copy.deepcopy(wIn)
    b= bIn
    for i in range(numOfIter):
        djdw, djdb = gradient(x,y,w,b,lmd)
        wInit = w-(a*djdw)
        bInit = b-(a*djdb)
        w = wInit
        b = bInit
        cost.append(costFunction(x,y,w,b))
        history.append([w,b])
        if i% math.ceil(numOfIter/10) == 0:
            print(f'{i}th Iteration Cost: {cost[i]} Parameters: w = {w}, b = {b}')
    return w,b,cost

np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
w, b = gradientDescent(10000,X_tmp,y_tmp,w_tmp,b_tmp,0.01,lambda_tmp)
print(w,b)