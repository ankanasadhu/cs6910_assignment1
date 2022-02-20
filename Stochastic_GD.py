import Model 
import numpy as np
from keras.datasets import fashion_mnist
def cross_entropy(out):
        loss = -1 *  np.log(out)
        return loss
                       

def gradient_descent():
    ann.output_layer.weights -= (ann.eta * ann.output_layer.g_weights)
    ann.output_layer.biases -= (ann.eta * ann.output_layer.g_biases)
    for x in (ann.layers):
        x.weights -= (ann.eta * x.g_weights)
        x.biases -= (ann.eta * x.g_biases)

def train():
    i = 0
    for img in ann.input_layer:
        true_y = np.zeros(10)
        true_y[ann.real_outputs[i]] = 1
        activation = np.ndarray.flatten(img)   
        op = ann.forwardprop(activation) 
        if i==0 :
            print("LOSS=",cross_entropy(op[ann.real_outputs[i]]))
        i += 1
        ann.backprop(op,img,true_y)
        gradient_descent()
   
    
def reinitialize():
    ann.output_layer.g_weights = np.zeros(ann.output_layer.weights.shape)
    ann.output_layer.g_biases = np.zeros(ann.output_layer.biases.shape)
    for x in ann.layers:
        x.g_biases = np.zeros(x.biases.shape)
        x.g_weights = np.zeros(x.weights.shape)

(train_x, train_Y), (test_x, test_Y) = fashion_mnist.load_data()


ann = Model.FFNN(train_x[:100]/255, train_Y, 0.001)
ann.add_hidden_layer(200)
ann.add_hidden_layer(100)
epochs = 1000
for x in range(epochs):
    train()
    reinitialize()
