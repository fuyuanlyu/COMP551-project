import numpy as np
def ReLu(z):
    '''
    Input: a np.array
    '''
    z[z<0]=0
    return z

def ReLu_derivative(a):
    return 0 if a <=0 else 1

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

def softmax_cross_entropy(z,y):
    