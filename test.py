import numpy as np
import matplotlib.pyplot as plot


def sigmoid(x, derivative=False):
    y = 1 / (1+np.exp(-x))
    if derivative:
        return y * (1-y)
    else:
        return y

def tanh(x, derivative = False):
    a = 1.7
    b = 0.667
    d = 0.
    if derivative:
        d = 2 * a * b * np.exp(-b*x) / ((1+np.exp(-b*x))**2)
    else:
        d = 2*a / (1+np.exp(-b*x)) - a
    return d


class Layer:
    def __init__(self, num_inputs, num_outputs):
        self.weights = np.random.random((num_inputs+1, num_outputs))
        print(self.weights)
        self.tmp_w_delta = np.zeros((num_inputs+1, num_outputs))
        pass
    
    def calc(self, inputs):
        self.tmp_inputs = inputs
        sum = np.dot(inputs, self.weights[1:,])
        sum -= self.weights[0] #bias
        self.tmp_output = sum
        return tanh(sum)

    def calc_error_delta(self, fwd_layer):
        self.tmp_error_delta = tanh(self.tmp_output, True) * (fwd_layer.tmp_error_delta@fwd_layer.weights[1:,].T)
        pass
    def calc_weights_delta(self):
        self.tmp_inputs = np.matrix(self.tmp_inputs)
        self.tmp_error_delta = np.matrix(self.tmp_error_delta)
        #print(self.tmp_inputs.T.shape, self.tmp_error_delta.shape)
        self.tmp_w_delta = 0.05 * (self.tmp_inputs.T@self.tmp_error_delta)
        tmp_bias_delta = 0.05 * -1 * self.tmp_error_delta
        self.tmp_w_delta = np.vstack((tmp_bias_delta, self.tmp_w_delta))
        pass
    def apply(self):
        self.weights += self.tmp_w_delta

np.random.seed(1)
x = np.array([0,0,0,1,1,0,1,1]).reshape((4,2))
y = np.array([0,1,1,0]).reshape((4,1))
layer1 = Layer(2, 2)
layer2 = Layer(2, 1)
'''
layer1.weights = np.array([[ 3.2026776 ,  3.37964631],[-5.98021874,  6.07068957],[ 5.76507326, -6.21467123]]).reshape((3,2))
layer2.weights = np.array([[4.71916944], [9.58508582], [9.50748812]]).reshape((3,1))
'''
epochs = 0
epochs_arr = []
errors = 0
errors_arr = []
plot.xlabel('epochs')
plot.ylabel('error')
while 1:
    errors = 0
    for xi, yi in zip(x,y):
        o = layer1.calc(xi)
        o = layer2.calc(o)
        
        error = yi - o
        errors += error**2
        layer2.tmp_error_delta = tanh(layer2.tmp_output, True)*error
        layer1.calc_error_delta(layer2)
        layer2.calc_weights_delta()
        layer1.calc_weights_delta()
        layer2.apply()
        layer1.apply()
    epochs+=1
    if(epochs %100 == 0):
        epochs_arr.append(epochs)
        errors_arr.append(errors)
        print(epochs, errors)
    if errors  < 0.001:
        break

print(errors)
plot.plot(epochs_arr, errors_arr)
plot.savefig('result.png')
for xi, yi in zip(x,y):
    o = layer1.calc(xi)
    o = layer2.calc(o)
    error = yi - o
    print(xi, yi, o)

print(layer1.weights)
print(layer2.weights)