import math, copy
import numpy as np
import matplotlib.pyplot as plt

def computeCost(x,y,w,b):
    m = len(x)
    fwb = 0
    for i in range(m):
        fwbOld = ((w*x[i]+b)-y[i])**2
        fwb = fwb+fwbOld
    cost = (1/(2*m))*fwb
    return cost

def derivative(x,y,w,b):
    m = len(x)
    f_wb = 0
    sum_dj_db=0
    sum_dj_dw=0
    for i in range(m):
        f_wb = w*x[i]+b
        sum_dj_dw_tmp = (f_wb-y[i])*x[i]
        sum_dj_db_tmp = (f_wb-y[i])
        sum_dj_dw+=sum_dj_dw_tmp
        sum_dj_db+=sum_dj_db_tmp
    dj_dw = (1/m)*sum_dj_dw
    dj_db = (1/m)*sum_dj_db
    return dj_dw, dj_db

def gradientDescent(x, y, wIn, bIn, a, numOfIter):
    w = copy.deepcopy(wIn)
    b = bIn
    w = wIn

    costHistory = []
    paramHistory = []
    for i in range(numOfIter):
        dj_dw, dj_db = derivative(x,y,w,b)
        wTemp = w-(a*dj_dw)
        bTemp = b-(a*dj_db)
        w = wTemp
        b = bTemp
        costHistory.append(computeCost(x,y,w,b))
        paramHistory.append([w,b])
        if i% math.ceil(numOfIter/10) == 0:
            print(f"Iteration {i: 4}: Cost {costHistory[i]: 0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w,b,costHistory,paramHistory
x_train=np.array([1.0, 2.0])
y_train=np.array([300.0, 500.0])
w = 0
b= 0
iter = 1000
a = 0.001
w, b, costHistory, paramHistory = gradientDescent(x_train,y_train,w,b,a,iter)
print(w,b)
