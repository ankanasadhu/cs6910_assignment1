import numpy as np
from keras.datasets import fashion_mnist

def Sigmoid(x): # here input is a vector
    return 1 / (1 + (np.e) ** (-x)) 

def exp (y) :
	return (np.e)**y

def Softmax(x):
    return exp(x) / np.sum(exp(x))

def Diff_Sigmoid(x) :
    return np.multiply(Sigmoid(x), (Sigmoid(x) - np.ones(x.shape)))

def Reshape (vector):
    return vector.reshape(vector.shape[0],1)

class Output_Layer:
    def __init__(self, neurons, prev_layer_neurons):
        self.neurons = neurons
        self.weights = np.random.rand(neurons, prev_layer_neurons) - 0.5
        self.biases = np.random.rand(neurons) - 0.5
        self.g_weights = np.zeros(self.weights.shape)
        self.g_biases = np.zeros(self.biases.shape)
    
    def change_weights(self, prev_layer_neurons):
        self.weights = np.random.rand(self.neurons, prev_layer_neurons)
        self.g_weights = np.zeros(self.weights.shape)
    
    def forward(self, input_):
        self.pre_activation = np.dot(self.weights, input_) + self.biases
        self.post_activation = Softmax(self.pre_activation)
        return self.post_activation
    
    def backward(self, output_true, output_pred, prev_post_activation, prev_pre_activation):
        self.grad_a_Ltheta = -(np.subtract(output_true, output_pred))
        self.grad_W_Ltheta = np.matmul(Reshape(self.grad_a_Ltheta), np.transpose(Reshape(prev_post_activation)))
        self.grad_b_Ltheta = self.grad_a_Ltheta
        self.grad_prev_post_activation_Ltheta = np.ndarray.flatten(np.matmul(np.transpose(self.weights), Reshape(self.grad_a_Ltheta)))
        self.dg = Diff_Sigmoid(prev_pre_activation)
        self.grad_a_Ltheta = np.ndarray.flatten(np.multiply(Reshape(self.grad_prev_post_activation_Ltheta), Reshape(self.dg)))
        return self.grad_a_Ltheta


class Hidden_Layer:
    def __init__(self, neurons, prev_layer_neurons):
        self.neurons = neurons
        self.weights = np.random.rand(neurons, prev_layer_neurons) - 0.5
        self.biases = np.random.rand(neurons) - 0.5
        self.g_weights = np.zeros(self.weights.shape)
        self.g_biases = np.zeros(self.biases.shape)

    def forward(self, input_):
        self.pre_activation = np.dot(self.weights, input_) + self.biases
        self.post_activation = Sigmoid(self.pre_activation)
        return self.post_activation

    def backward(self, next_grad_a_LTheta, prev_pre_activation, prev_post_activation):
        
        self.grad_W_Ltheta = np.matmul(Reshape(next_grad_a_LTheta), np.transpose(Reshape(prev_post_activation)))
        self.grad_b_Ltheta = next_grad_a_LTheta
        self.grad_prev_post_activation_Ltheta= np.ndarray.flatten(np.dot(np.transpose(self.weights), Reshape(next_grad_a_LTheta)))
        self.dg = Diff_Sigmoid(prev_pre_activation)
        self.grad_a_Ltheta = np.ndarray.flatten(np.multiply(Reshape(self.grad_prev_post_activation_Ltheta), Reshape(self.dg)))
        return self.grad_a_Ltheta

# prev_grad_a_Ltheta refers to the i+1 th layer gradient whereas prev_post_activation and prev_pre_activation refers to i-1 th layer pre and post activation fns
        
class FFNN:
    def __init__(self, input_, outputs_, learning_rate):
        self.real_outputs = outputs_
        self.input_layer = input_ # this is the input vector
        self.ip_dim = input_.shape[1] * input_.shape[2]
        self.layers = []
        self.add_default_layers(self.ip_dim, 10)
        self.eta = learning_rate

    def add_default_layers(self, input_dim, output_dim):
        self.output_layer = Output_Layer(output_dim, input_dim)

    def add_hidden_layer(self, no_of_neurons):
        if(len(self.layers) == 0):
            new_layer = Hidden_Layer(no_of_neurons, self.ip_dim)
        else:
            new_layer = Hidden_Layer(no_of_neurons, self.layers[-1].neurons)
        self.layers.append(new_layer)
        # changing the weights of output layer each time a new hidden layer is added
        self.output_layer.change_weights(self.layers[-1].neurons)
    
    def train(self):
        i = 0
        for img in self.input_layer:
            true_y = np.zeros(10)
            true_y[self.real_outputs[i]] = 1
            activation = np.ndarray.flatten(img)
            
            for x in range(len(self.layers)):
                activation = self.layers[x].forward(activation)
            
            op = self.output_layer.forward(activation)

            loss = -1 *  np.log(op[self.real_outputs[i]])
            if(i == 0):
                print("LOSS=",loss)
            i += 1
          
           

                

    

(train_x, train_Y), (test_x, test_Y) = fashion_mnist.load_data()

ann = FFNN(train_x[:3]/255, train_Y, 0.001)
ann.add_hidden_layer(200)
ann.add_hidden_layer(100)
