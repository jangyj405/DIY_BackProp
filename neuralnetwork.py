from layer import Layer
from activator import *
from matplotlib import pyplot as plot

class NeuralNetwork:
    def __init__(self, outputs, activator, from_file=""):
        self.layers = []
        self.layer_sizes = []
        self.outputs = outputs
        self.activator = activator
        if(from_file != ""):
            #load from file
            pass
        pass

    def add_layer(self, inputs):
        self.layer_sizes.append([inputs, self.outputs])
        if len(self.layer_sizes) == 1:
            pass
        else:
            self.layer_sizes[-2][1] = self.layer_sizes[-1][0]
        pass
    
    def rearrange(self):
        if len(self.layer_sizes) == 0:
            print("No layers")
        self.layers = [Layer(self.activator,(x,y)) for (x,y) in self.layer_sizes]
        pass

    def calc(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer.calc(x)
        return x
    
    def fit(self, error):
        last_layer = self.layers[-1]
        last_layer.tmp_error_delta = last_layer.activator(last_layer.tmp_output, True)*error
        for i in reversed(range(len(self.layers)-1)):
            self.layers[i].calc_error_delta(last_layer)
            last_layer = self.layers[i]
        for layer in self.layers:
            layer.calc_weights_delta()
            layer.apply()


if __name__ == "__main__":
    np.random.seed(1)
    x = np.array([0,0,0,1,1,0,1,1]).reshape((4,2))
    y = np.array([0,1,1,0]).reshape((4,1))

    nn = NeuralNetwork(1, activator=tanh)
    nn.add_layer(2)
    nn.add_layer(2)
    nn.rearrange()
    epochs = 0
    errors = 0

    arr_epochs = []
    arr_errors = []
    plot.xlabel("epochs")
    plot.ylabel('errors')
    while 1:
        errors = 0
        for xi, yi in zip(x,y):
            o = nn.calc(xi)
            error = yi - o
            errors += sum(error**2)
            nn.fit(error)
        errors /= y.shape[0]
        epochs+=1
        arr_epochs.append(epochs)
        arr_errors.append(errors)
        if(epochs %10 == 0):
            print(epochs, errors)
        if errors  < 0.001:
            break

    print(epochs, errors)
    plot.plot(arr_epochs, arr_errors)
    plot.savefig('result')

    for xi, yi in zip(x,y):
        o = nn.calc(xi)
        error = yi - o
        print(xi, yi, o)
