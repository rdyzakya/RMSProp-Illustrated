import numpy as np

def mse_loss(x, y, m, c):
    return (y - (m*x + c))**2

################################

def sigmoid(x, m, c):
    return 1 / (1 + np.exp(-(m*x + c)))

def bce_loss(x, y, m, c):
    o = sigmoid(x, m, c)
    return -(y*np.log(o) + (1-y)*np.log(1-o))