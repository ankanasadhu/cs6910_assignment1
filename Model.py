import numpy as np
from Optimizer import momentum_gradient_descent, momentum_or_nag_gd, sgd_or_batch_gd, rmsprop, adam

def cross_entropy(out):
    loss = -1 *  np.log(out + 0.0001)
    return loss

def squared_error(y_true, y_pred) :
    loss=0
    l=len(y_true)
    for i in range (l):
     loss+= np.square(y_true[i]-y_pred[i])/l
    return loss

def Sigmoid(x): # here input is a vector
    return 1 / (1 + (np.e) ** (-x)) 

def exp (y) :
	return (np.e)**y

def Softmax(x):
    return exp(x) / np.sum(exp(x))

def Diff_Sigmoid(x) :
    return np.multiply(Sigmoid(x), ( np.ones(x.shape) - Sigmoid(x) ))

def Reshape (vector):
    return vector.reshape(vector.shape[0],1)

def Relu(x) :
    x= x / np.max(x)
    y= np.maximum(0,x)
    #print (y)
    return y

def Diff_Relu (x):
    y= np.zeros(x.shape)
    for i in range (len (x)): 
        y[i]=1 if x[i]>=0 else 0
    return y 

def Tanh (x):
    return (exp(x) - exp (-x))/(exp(x) + exp (-x))

def Diff_Tanh (x) :
    return (np.ones(x.shape) - np.square(Tanh(x)))

class Output_Layer:
    def __init__(self, neurons, prev_layer_neurons):
        self.neurons = neurons
        self.weights = np.random.rand(neurons, prev_layer_neurons) - 0.5
        self.biases = np.random.rand(neurons) - 0.5
        self.g_weights = np.zeros(self.weights.shape)
        self.g_biases = np.zeros(self.biases.shape)
         # added for momentum gradient descent
        self. prev_v_w= np.zeros(self.weights.shape)
        self.prev_v_b=  np.zeros(self.biases.shape)
        #added for adam
        self.m_w= np.zeros(self.weights.shape)
        self.m_b=  np.zeros(self.biases.shape)
    
    def change_weights(self, prev_layer_neurons):
        self.weights = np.random.rand(self.neurons, prev_layer_neurons)
        self.g_weights = np.zeros(self.weights.shape)
        self. prev_v_w= np.zeros(self.weights.shape)
        self.m_w= np.zeros(self.weights.shape)
    
    def forward(self, input_):
        self.pre_activation = np.dot(self.weights, input_) + self.biases
        self.post_activation = Softmax(self.pre_activation)
        return self.post_activation
    
    def backward(self, output_true, output_pred, prev_post_activation, prev_pre_activation,gamma,isnag, post_activation_function, loss_function='cross_ent'):
        if(loss_function == "cross_ent"):
            self.grad_a_Ltheta = -(np.subtract(output_true, output_pred))
        if(loss_function == "sq_err"):
            one=np.ones(output_pred.shape)
            self.grad_a_Ltheta =-(output_true- output_pred)*(output_pred)*(one-output_pred)
        #print("True=",output_true)
        #print("Pred=",output_pred)
        #print("grad A", self.grad_a_Ltheta)
        # print("grad A", prev_post_activation)
        self.grad_W_Ltheta = np.matmul(Reshape(self.grad_a_Ltheta), np.transpose(Reshape(prev_post_activation)))
        #print("grad W_output=", self.grad_W_Ltheta)
        self.grad_b_Ltheta = self.grad_a_Ltheta
        #print("grad B out_put=", self.grad_b_Ltheta)
        weight= self.weights-gamma*isnag*self.prev_v_w 
        #print("Output Layer",self.weights)
        #print("Output Layer",weight)
        self.grad_prev_post_activation_Ltheta = np.ndarray.flatten(np.matmul(np.transpose(weight), Reshape(self.grad_a_Ltheta)))
        if post_activation_function=="relu" :
            self.dg=Diff_Relu(prev_pre_activation)
        if post_activation_function=="logistic" :
            self.dg = Diff_Sigmoid(prev_pre_activation)
        elif post_activation_function=="tanh":
            self.dg = Diff_Tanh(prev_pre_activation)
        # print("grad_hk-1=",self.grad_prev_post_activation_Ltheta)
        # print("g'(a)=",self.dg)
        self.grad_a_Ltheta = np.ndarray.flatten(np.multiply(Reshape(self.grad_prev_post_activation_Ltheta), Reshape(self.dg)))
        return self.grad_a_Ltheta


class Hidden_Layer:
    def __init__(self, neurons, prev_layer_neurons):
        self.neurons = neurons
        self.weights = np.random.rand(neurons, prev_layer_neurons) - 0.5
        self.biases = np.random.rand(neurons) - 0.5
        self.g_weights = np.zeros(self.weights.shape)
        self.g_biases = np.zeros(self.biases.shape)
        # added for momentum gradient descent
        self. prev_v_w= np.zeros(self.weights.shape)
        self.prev_v_b=  np.zeros(self.biases.shape)
        # added for adam
        self.m_w= np.zeros(self.weights.shape)
        self.m_b=  np.zeros(self.biases.shape)

    def forward(self, input_, post_activation_function):
        self.pre_activation = np.dot(self.weights, input_) + self.biases
        if post_activation_function=="relu":
            self.post_activation= Relu(self.pre_activation)
        elif post_activation_function=="logistic":
            self.post_activation = Sigmoid(self.pre_activation)
        elif post_activation_function=="tanh":
            self.post_activation = Tanh(self.pre_activation)
        return self.post_activation

    def backward(self, grad_a_LTheta, prev_pre_activation, prev_post_activation,gamma,isnag, post_activation_function):
        
        self.grad_W_Ltheta = np.matmul(Reshape(grad_a_LTheta), np.transpose(Reshape(prev_post_activation)))
        #print("grad G=",self.grad_W_Ltheta)
        self.grad_b_Ltheta = grad_a_LTheta
        #print("gradB",self.grad_b_Ltheta)
        # print("h", prev_post_activation)
        weight= self.weights-gamma*isnag*self.prev_v_w
        #print("Hidden Layer",gamma*isnag*self. prev_v_w*1000)
        self.grad_prev_post_activation_Ltheta= np.ndarray.flatten(np.dot(np.transpose(weight), Reshape(grad_a_LTheta)))
        if post_activation_function=="relu" :
            self.dg=Diff_Relu(prev_pre_activation)
        if post_activation_function=="logistic" :
            self.dg = Diff_Sigmoid(prev_pre_activation)
        elif post_activation_function=="tanh":
            self.dg= Diff_Tanh(prev_pre_activation)
        # print("grad_hk-1=",self.grad_prev_post_activation_Ltheta)
        # print("g'(a)=",self.dg)
        self.prev_grad_a_Ltheta = np.ndarray.flatten(np.multiply(Reshape(self.grad_prev_post_activation_Ltheta), Reshape(self.dg)))
        return self.prev_grad_a_Ltheta

# prev_grad_a_Ltheta refers to the i+1 th layer gradient whereas prev_post_activation and prev_pre_activation refers to i-1 th layer pre and post activation fns
        
class FFNN:
    def __init__(self, learning_rate, var=0): # the arguments for passing data are not required
        # self.test_x = test_x
        # self.test_y = test_y
        # self.real_outputs = outputs_
        # self.input_layer = input_ # this is the input vector
        self.ip_dim = 784
        self.layers = []
        self.add_default_layers(self.ip_dim, 10)
        self.eta = learning_rate
        self.gamma= 0.9
        self.isnag= var
        self.eps=1e-8
        self.beta1=0.9
        self.beta2=0.999

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
    def forwardprop(self,activation, post_activation_function):
        
        for x in range(len(self.layers)):
            activation = self.layers[x].forward(activation, post_activation_function)
        op_ = self.output_layer.forward(activation)
        return op_
    def backprop (self,op,img,true_y, batchsize, post_activation_function, loss_function):
        self.prev_post_activation = self.layers[-1].post_activation
        self.prev_pre_activation = self.layers[-1].pre_activation          
        self.grad_a_Ltheta = self.output_layer.backward(true_y, op, self.prev_post_activation, self.prev_pre_activation,self.gamma,self.isnag, post_activation_function, loss_function)
        self.output_layer.g_weights += (1/ batchsize) * self.output_layer.grad_W_Ltheta # (10, 100)
        #print("Dw_sum=",self.output_layer.g_weights)
        self.output_layer.g_biases += (1/ batchsize) * self.output_layer.grad_b_Ltheta # (10, )

        for x in range(len(self.layers)-1,-1,-1):
            #print("Hidden Layer=",x)
            if x==0:
                post_activation = np.ndarray.flatten(img)
                pre_activation = np.zeros(post_activation.shape)
            else:
                pre_activation = self.layers[x-1].pre_activation # previous layer a(k-1) 
                post_activation = self.layers[x-1].post_activation # previous layer h(k-1)
                
            self.grad_a_Ltheta = self.layers[x].backward(self.grad_a_Ltheta, pre_activation, post_activation,self.gamma,self.isnag, post_activation_function)
            self.layers[x].g_weights += (1/ batchsize) * self.layers[x].grad_W_Ltheta              
            self.layers[x].g_biases += (1/ batchsize) * self.layers[x].grad_b_Ltheta
        
    # average over all the gradients by passing the batch size
    
    def train(self, train_, batch_size, data_x, data_y, epoch, opt='sgd', activation_func='logistic', loss_function='cross_ent'):
        i = 0
        count = 0
        avg_loss=0
        for img in data_x:
            true_y = np.zeros(10)
            true_y[data_y[i]] = 1
            activation = np.ndarray.flatten(img)   
            op = self.forwardprop(activation, activation_func)
            if(loss_function == 'cross_ent'):
                avg_loss += cross_entropy(op[data_y[i]])
            if(loss_function == 'sq_err'):
                avg_loss +=  squared_error(true_y, op)
            if(np.argmax(op) == data_y[i]):
                count += 1
            i += 1
            if(train_):
                self.backprop(op,img,true_y, batch_size, activation_func, loss_function) # computing the gradients for each image and adding them up
        acc = count / len(data_x)
        avg_loss = avg_loss / len(data_x)
        if(train_):
            if(opt == 'momentum'):
                self = momentum_gradient_descent(self)
            if(opt == 'sgd'):
                self = sgd_or_batch_gd(self, batch_size)
            if(opt == 'rmsprop'):
                self = rmsprop(self)
            if(opt == 'adam'):
                self = adam(self, epoch)
            if(opt == 'nag'):
                self = momentum_or_nag_gd(self)
            if(opt == 'nadam'):
                self = adam(self, epoch)
            self.reinitialize()
        return avg_loss, acc
    
    def reinitialize(self):
        self.output_layer.g_weights = np.zeros(self.output_layer.weights.shape)
        self.output_layer.g_biases = np.zeros(self.output_layer.biases.shape)
        for x in self.layers:
            x.g_biases = np.zeros(x.biases.shape)
            x.g_weights = np.zeros(x.weights.shape)
    
