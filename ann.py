from email.mime import image
import numpy as np
from keras.datasets import fashion_mnist

def Sigmoid(x): # here input is a vector
    return 1 / (1 + (np.e) ** (-1 * x)) 

def exp (y) :
	return (np.e)**y

def Softmax(x):
    return exp(x) / np.sum(exp(x))

class Hidden_Layer:
    def __init__(self, neurons, prev_layer_neurons):
        self.neurons = neurons
        self.weights = np.random.rand(neurons, prev_layer_neurons) - 0.5
        self.biases = np.random.rand(neurons) - 0.5
        # self.g_weights = np.zeros(self.weights.shape)
        # self.g_biases = np.zeros(self.biases.shape)

    def forward(self, input_):
        self.pre_activation = np.dot(self.weights, input_) + self.biases
        self.post_activation = Sigmoid(self.pre_activation)
        return self.post_activation

    def backward(self):
        pass

class Output_Layer:
    def __init__(self, neurons, prev_layer_neurons) -> None:
        self.neurons = neurons
        self.weights = np.random.rand(neurons, prev_layer_neurons) - 0.5
        self.biases = np.random.rand(neurons) - 0.5
    
    def change_weights(self, prev_layer_neurons):
        self.weights = np.random.rand(self.neurons, prev_layer_neurons)
    
    def forward(self, input_):
        self.pre_activation = np.dot(self.weights, input_) + self.biases
        self.post_activation = Softmax(self.pre_activation)
        return self.post_activation
    
    def backward(self):
        pass

class FFNN:
    def __init__(self, input_, outputs_):
        self.real_outputs = outputs_  # true y value
        self.input_layer = input_ # this is the input vector
        self.ip_dim = input_.shape[1] * input_.shape[2]
        self.layers = []
        self.output_layer = Output_Layer(self.ip_dim, 10)

    def add_hidden_layer(self, no_of_neurons):
        if(len(self.layers) == 0):
            new_layer = Hidden_Layer(no_of_neurons, self.ip_dim)
        else:
            new_layer = Hidden_Layer(no_of_neurons, self.layers[-1].neurons)
        self.layers.append(new_layer)
        # changing the weights of output layer each time a new hidden layer is added
        self.output_layer.change_weights(self.layers[-1].neurons)
    

