Assignment 1: Neural Network model development

This project is created to implement feed forward neural network and use gradient descent algorithm (and its different variants such as momentum, nesterov accelerated, adam, nadam,rmsprop and stochastic gradient descent).
Each image must be classified into one of 10 labels.
We have use wandb data visualisation to generate graphs and reports of the different methods and produce insightful observations.

Libraries used:

We have used numpy for all the mathematical calculations in forward propagation, back propagation algorithm and loss function calculations.
Scikit learn library was used to generate the confusion matrix which was converted into a dataframe using pandas library.
Seaborn and matplotlib libraries were used for plotting the confusion matrix.
Keras and tensorflow was used for getting the fashion mnist dataset.
Pickle was used to save the best neural network model obtained during training.

Installations:

* pip installment - <requirements.txt> was used for the purpose of installments.

How to use?

1. Start execution from train.py. call the train() method and pass to run the required values to construct the neural network
2. Depending on the values of optimiser, number of neurons in each layer, error functions the code will construct the neural network.

How to change the number of layers used, the number of neurons in each layer and the type of activation function used in each layer?
1. Depending on the layer to be added or removed call the add_hidden_layer function the required numbe of times from train.py in method run()
2. If add_hidden_layer() is called twice from method run() in trin.py it implies two hidden layers will be added.
3. add_hidden_layer(layer_1) here, layer_1 is the number of neurons in the first layer.

How to add a new optimisation algorithm to the existing code?

The current code structure is fexible enough to accodomodate new optimisation algorithms.
1. by defining new functions to the Optimizer.py file for the required optimisation algorithm.
2. calling the same function from train() function of train.py will 

Forward Propagation Procedure:
1.  forwardprop() function of class FFNN is called which further  calls the layers[].forward() and output_layer.forward()
2.  layers[].forward() is executed first which computes the layers[].pre_activation and layers[].post_activation matrices of the HIDDEN LAYER using the mentioned activation function(logistic/tanh/relu)
3.  The above step is followed by execution of output_layer.forward() and output_layer.pre_activation and output_layer.post_activation matrices of the OUTPUT LAYER using the softmax activation function to calculate the predicted output label.
* Calculates the pre-activation and the post activation matrices with the previously computed weights and biases for the dataset given.

Backward Propagation Procedure:
1. backwardprop() function of class FFNN is called which further calls output_layer.backward() and layers[].backward()
2. output_layer.backprop is executed first which computes:
  i) the grad_a_Ltheta matrix that is gradient of L(theta) w.r.t output layer pre-activation matrix.This is used to calculate gradient of Ltheta w.r.t weights and biases separately.
  ii)then, grad_a_Ltheta matrix is updated to gradient of L(theta) w.r.t last hidden layer pre-activation matrix and post-activation matrix.
3. layers[].backprop is executed next which computes:
  i) the updated grad_a_Ltheta is used to calculate gradient of Ltheta w.r.t weights and biases separately.
  ii)then, grad_a_Ltheta matrix is updated to gradient of L(theta) w.r.t (k-1)th [ where k iterates from=Last hidden layer to 1st hidden layer] hidden layer pre-activation matrix and post-activation matrix.
* Calculates gradient of L(theta) w.r.t. weights and biases for all the layers (hidden layers and output layer).

Construction of Neural Network: 
* The neural network is first created for an input layer and output layer with zero hidden layers.
*  The training dataset is split into training and 10% of the training dataset is kept separate as testing.
* Depending on the number of times add_hidden_layer function of FNN is called that many number of hiddenlayers is added to the neural network.
* Depending on the number of neurons passed while calling add_hidden_layer i.e. add_hidden_layer(layer_1) layer_1 stores the number of neurons in layer_1.

Program Flow
-->execution starts from train.py file --> method train()
--> then method run() is called from train() of train.py
--> run() creates an instance of class FFNN of Model.py of name ann
--> run() calls forwardprop() and backprop() of class FFNN of Model.py 
--> then control returns to train() of train() and then calls forwardprop() of class FFNN to compute results for validation
--> then similarly again forwardprop() of class FFNN is called to compute results for test.


Sweep:
The following combinations were used to create the yaml file for sweep to minimize validation loss and maximize validation accuracy:
* parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
 * optimizer:
    values: ["sgd", "adam", "momentum", "rmsprop", "nag"]
 * epochs:
    distribution: int_uniform
    min: 5
    max: 20
 * batch_size:
    distribution: int_uniform
    min: 50
    max: 600
* layer_1:
    values: [128, 254]
*  layer_2:
    values: [64, 32]
* layer_3:
    values: [32, 16]

Best Configuration:
* learning rate: 0.015465973439807984
* optimizer: adam 
* epoch: 20
* batch size:59
* activation: logistic


Link to Report:
https://wandb.ai/cs21m809_anukul/deep_learning-working_code/reports/Deep-Learning-Assignment-1-Report--VmlldzoxNjA3MjQz

Acknowledgements
* The entire project has been developed from the lecture slides of Prof. Mitesh Khapra, Indian Institute of Technology Madras: http://cse.iitm.ac.in/~miteshk/CS6910.html#schedule
* https://wandb.ai
* https://github.com/
                      
