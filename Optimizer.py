import Model 
import numpy as np
from keras.datasets import fashion_mnist

def momentum_gradient_descent(ann):
    v_w= ann.gamma * ann.output_layer.prev_v_w + ann.eta * ann.output_layer.g_weights
    ann.output_layer.weights -= v_w
    v_b=  ann.gamma * ann.output_layer.prev_v_b + ann.eta * ann.output_layer.g_biases
    ann.output_layer.biases -= v_b
    ann.output_layer.prev_v_w =v_w
    ann.output_layer.prev_v_b= v_b
    for x in (ann.layers):
        v_w= ann.gamma * x.prev_v_w + ann.eta * x.g_weights
        #print("GRAD G",x.g_weights)
        x.weights -= v_w
        v_b=  ann.gamma * x.prev_v_b + ann.eta * x.g_biases
        x.biases -= v_b
        x.prev_v_w=v_w
        #print("update",x.prev_v_w)
        x.prev_v_b=v_b
    return ann

def momentum_or_nag_gd(ann):
    v_w= ann.gamma * ann.output_layer.prev_v_w + ann.eta * ann.output_layer.g_weights
    ann.output_layer.weights -= v_w
    v_b=  ann.gamma * ann.output_layer.prev_v_b + ann.eta * ann.output_layer.g_biases
    ann.output_layer.biases -= v_b
    ann.output_layer.prev_v_w =v_w
    ann.output_layer.prev_v_b= v_b
    for x in (ann.layers):
        v_w= ann.gamma * x.prev_v_w + ann.eta * x.g_weights
        x.weights -= v_w
        v_b=  ann.gamma * x.prev_v_b + ann.eta * x.g_biases
        x.biases -= v_b
        x.prev_v_w=v_w
        x.prev_v_b=v_b
    return ann

def sgd_or_batch_gd(ann, n):
    ann.output_layer.weights -= (ann.eta * ann.output_layer.g_weights)
    ann.output_layer.biases -= (ann.eta * ann.output_layer.g_biases)
    for x in (ann.layers):
        x.weights -= (ann.eta * x.g_weights)
        x.biases -= (ann.eta * x.g_biases)
    return ann

def rmsprop(ann):
    
    ann.output_layer.prev_v_w= ann.beta1 * ann.output_layer.prev_v_w + (1-ann.beta1) * (ann.output_layer.g_weights ** 2)
    ann.output_layer.weights -= (ann.eta/np.sqrt(ann.output_layer.prev_v_w + ann.eps))* ann.output_layer.g_weights
    ann.output_layer.prev_v_b=  ann.beta1 * ann.output_layer.prev_v_b + (1-ann.beta1) *  (ann.output_layer.g_biases ** 2)
    ann.output_layer.biases -= (ann.eta/np.sqrt(ann.output_layer.prev_v_b + ann.eps))* ann.output_layer.g_biases
    for x in (ann.layers):
        x.prev_v_w= ann.beta1 * x.prev_v_w + (1-ann.beta1) * (x.g_weights ** 2)
        x.weights -= (ann.eta/np.sqrt(x.prev_v_w + ann.eps))* x.g_weights
        x.prev_v_b= ann.beta1 * x.prev_v_b + (1-ann.beta1) * (x.g_biases** 2)
        x.biases -= (ann.eta/np.sqrt(x.prev_v_b + ann.eps))* x.g_biases
    return ann

def adam(ann, epoch):
    ann.output_layer.m_w= ann.beta1*ann.output_layer.m_w + (1-ann.beta1)* ann.output_layer.g_weights
    ann.output_layer.m_b= ann.beta1*ann.output_layer.m_b + (1-ann.beta1)* ann.output_layer.g_biases
    
    ann.output_layer.prev_v_w=  ann.beta2*ann.output_layer.prev_v_w+ (1-ann.beta2)* (ann.output_layer.g_weights **2)
    ann.output_layer.prev_v_b=  ann.beta2*ann.output_layer.prev_v_b+ (1-ann.beta2)* (ann.output_layer.g_biases **2)

    m_w_hat=ann.output_layer.m_w/(1- (ann.beta1 ** (epoch+1)))
    m_b_hat=ann.output_layer.m_b/(1- (ann.beta1 ** (epoch+1)))

    v_w_hat= ann.output_layer.prev_v_w/(1- (ann.beta2 ** (epoch+1)))
    v_b_hat= ann.output_layer.prev_v_b/(1- (ann.beta2 ** (epoch+1)))

    ann.output_layer.weights -= (ann.eta/np.sqrt(v_w_hat + ann.eps))* m_w_hat
    ann.output_layer.biases -= (ann.eta/np.sqrt(v_b_hat + ann.eps))* m_b_hat

    for x in (ann.layers):
        x.m_w= ann.beta1*x.m_w + (1-ann.beta1)* x.g_weights
        x.m_b= ann.beta1*x.m_b + (1-ann.beta1)* x.g_biases

        x.prev_v_w=  ann.beta2*x.prev_v_w+ (1-ann.beta2)* (x.g_weights **2)
        x.prev_v_b=  ann.beta2*x.prev_v_b+ (1-ann.beta2)* (x.g_biases **2)

        m_w_hat=x.m_w/(1- (ann.beta1 ** (epoch+1)))
        m_b_hat=x.m_b/(1- (ann.beta1 ** (epoch+1)))

        v_w_hat= x.prev_v_w/(1- (ann.beta2 ** (epoch+1)))
        v_b_hat= x.prev_v_b/(1- (ann.beta2 ** (epoch+1)))

        x.weights -= (ann.eta/np.sqrt(v_w_hat + ann.eps))* m_w_hat
        x.biases -= (ann.eta/np.sqrt(v_b_hat + ann.eps))* m_b_hat
    return ann

# def train(ann,optimizer):
#     i = 0
#     avg_loss=0
#     for img in ann.input_layer:
#         true_y = np.zeros(10)
#         true_y[ann.real_outputs[i]] = 1
#         activation = np.ndarray.flatten(img)   
#         op = ann.forwardprop(activation) 
#         avg_loss+= cross_entropy(op[ann.real_outputs[i]])
#         i += 1
#         ann.backprop(op,img,true_y)
#         if optimizer=="sgd":
#             sgd_or_batch_gd(ann)
    
#     if optimizer=="batch":
#         sgd_or_batch_gd(ann)
#     elif optimizer=="momentum" or optimizer=="nag":
#         momentum_or_nag_gd(ann)
    
#     reinitialize(ann)

# def reinitialize(ann):
#     ann.output_layer.g_weights = np.zeros(ann.output_layer.weights.shape)
#     ann.output_layer.g_biases = np.zeros(ann.output_layer.biases.shape)
#     for x in ann.layers:
#         x.g_biases = np.zeros(x.biases.shape)
#         x.g_weights = np.zeros(x.weights.shape)




