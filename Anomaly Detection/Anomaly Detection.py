import numpy as np

x = np.load("Data/X_part2.npy")
x_val = np.load("Data/X_val_part2.npy")
y_val = np.load("Data/y_val_part2.npy")

def estimateGaussian(x):
    m, n = x.shape

    mu = 1/m * np.sum(x,axis=0)
    var = 1/m * np.sum((x-mu)**2,axis = 0)
    return mu,var

def multivariateGaussian(x,mu,var):
    k = len(mu)
    if var.ndim == 1:
        var = np.diag(var)

    x = x-mu
    p = (2*np.pi)**(-k/2)*np.linalg.det(var)**(-0.5)*np.exp(-0.5*np.sum(np.matmul(x,np.linalg.pinv(var))*x,axis=1))

    return p

def selectThreshold(y,x):
    bestEpsilon = 0
    bestF1 = 0
    f1 = 0

    stepSize = (max(x)-min(x))/1000

    for ep in np.arange(min(x),max(x),stepSize):
        prediction = (x<ep)
        tp = np.sum((prediction == 1)&(y == 1))
        fn = np.sum((prediction == 0)&(y == 1))
        fp = np.sum((prediction == 1)&(y == 0))
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        f1 = 2*prec*rec/(prec+rec)

        if f1>bestF1:
            bestF1 = f1
            bestEpsilon = ep
    
    return bestEpsilon, bestF1

mu, var = estimateGaussian(x)
p1 = multivariateGaussian(x,mu,var)
p2 = multivariateGaussian(x_val,mu,var)
epsilon, f1 = selectThreshold(y_val,p2)

print(epsilon,f1,sum(p1<epsilon))
