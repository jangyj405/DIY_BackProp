import numpy as np

def sigmoid(x, derivative=False):
    y = 1 / (1+np.exp(-x))
    if derivative:
        return y * (1-y)
    else:
        return y
        
def tanh(x, derivative = False):
    a = 1#1.7
    b = 2#0.667
    d = 0.
    if derivative:
        d = 2 * a * b * np.exp(-b*x) / ((1+np.exp(-b*x))**2)
    else:
        d = 2*a / (1+np.exp(-b*x)) - a
    return d