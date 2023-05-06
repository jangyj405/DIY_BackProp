import numpy as np
from activator import *

class Layer:
    def __init__(self, activator, size = (1,1), weights = None):
        self.activator = activator
        if weights == None:
            inputs, outputs = size
            self.weights = np.random.random((inputs + 1, outputs))
        else:
            self.weights = weights
        self.tmp_w_delta = np.zeros(self.weights.shape)
        pass
    def calc(self, inputs):
        self.tmp_inputs = inputs
        sum = np.dot(inputs, self.weights[1:,])
        sum -= self.weights[0] #bias
        self.tmp_output = sum
        return self.activator(sum)

    def calc_error_delta(self, fwd_layer):
        #print(fwd_layer.tmp_error_delta.shape,fwd_layer.weights[1:,].T.shape)
        self.tmp_error_delta = self.activator(self.tmp_output, True) * (fwd_layer.tmp_error_delta@fwd_layer.weights[1:,].T)
        pass
    def calc_weights_delta(self, eta = 0.05, beta = 0.95):
        self.tmp_inputs = np.matrix(self.tmp_inputs)
        self.tmp_error_delta = np.matrix(self.tmp_error_delta)
        
        momentum = beta * self.tmp_w_delta[1:,]
        momentum_bias = beta * self.tmp_w_delta[0,]
        self.tmp_w_delta = momentum + eta * (self.tmp_inputs.T@self.tmp_error_delta)
        tmp_bias_delta = momentum_bias + eta * -1 * self.tmp_error_delta
        self.tmp_w_delta = np.vstack((tmp_bias_delta, self.tmp_w_delta))
        pass
    def apply(self):
        self.weights += self.tmp_w_delta



if __name__ == "__main__":
    _activator = tanh
    np.random.seed(1)
    x = np.array([0,0,0,1,1,0,1,1]).reshape((4,2))
    y = np.array([0,1,1,0]).reshape((4,1))
    layer1 = Layer(activator=_activator, size=(2, 2))
    layer2 = Layer(activator=_activator, size=(2, 1))

    #layer1.weights = np.array([[ 3.2026776 ,  3.37964631],[-5.98021874,  6.07068957],[ 5.76507326, -6.21467123]]).reshape((3,2))
    #layer2.weights = np.array([[4.71916944], [9.58508582], [9.50748812]]).reshape((3,1))

    epochs = 0
    errors = 0
    while 1:
        errors = 0
        for xi, yi in zip(x,y):
            o = layer1.calc(xi)
            o = layer2.calc(o)
            error = yi - o
            errors += error**2
            layer2.tmp_error_delta = layer2.activator(layer2.tmp_output, True)*error
            layer1.calc_error_delta(layer2)
            layer2.calc_weights_delta()
            layer1.calc_weights_delta()
            layer2.apply()
            layer1.apply()
        
        epochs+=1
        if(epochs %1000 == 0):
            print(epochs, errors)
        if errors  < 0.001:
            break

    print(epochs, errors)
    #plot.plot(epochs_arr, errors_arr)
    #plot.savefig('result.png')
    for xi, yi in zip(x,y):
        o = layer1.calc(xi)
        o = layer2.calc(o)
        error = yi - o
        print(xi, yi, o)

    print(layer1.weights)
    print(layer2.weights)
