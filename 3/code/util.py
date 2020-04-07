import numpy as np
def ReLu(z):
    '''
    Input: a np.array
    '''
    z[z<0]=0
    return z

def ReLu_derivative(a):
    d = np.zeros_like(a)
    d[a>0]=1
    return d

def sigmoid(z):
    '''
    Input: logit, a np.array
    '''
    return 1/(1+np.exp(-z))

def sigmoid_derivative(a):
    return np.multiply(a,(1.0-a))

def cross_entropy(true,prediction):
    return np.mean(true * np.log(prediction) + (1-true)*np.log(1-prediction))

def logsumexp(z):
    # z: C*N
    z_max = np.max(z,axis=0)[None,:]
    lse = z_max + np.log(np.sum(np.exp(z-z_max),axis=0))
    return lse

def one_hot(y):
    N = y.shape[0]
    C = 10
    y_hot = np.zeros((N,C))
    y_hot[np.arange(N),y] = 1
    return y_hot

def softmax_cross_entropy(z,y):
    Y = one_hot(y)
    nll = -np.sum(np.sum(z*Y,1)-logsumexp(z.T))
    return nll

def softmax(x):
    f = np.exp( (x - np.mean(x,axis=0))/np.std(x,axis=0) )  # shift values and normalize
    sm = f.T/f.sum(axis=1)
    return sm.T

