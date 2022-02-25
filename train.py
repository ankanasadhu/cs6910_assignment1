from pickletools import optimize
import wandb
from Model import FFNN
from keras.datasets import fashion_mnist
import numpy as np
# from Optimizer import momentum_gradient_descent, momentum_or_nag_gd, sgd_or_batch_gd, adam, rmsprop, reinitialize

(train_x, train_Y), (test_x, test_Y) = fashion_mnist.load_data()

# we have to train using the data and test for validation

# def logging(data_x, data_y, val_x, val_y, epochs, batch_size, learning_rate, optimizer, layer_1, layer_2, layer_3):
    
#     if(optimizer == "nag"):
#         ann = FFNN(learning_rate, 1) # create FFNN with all these values
#     else:
#         ann = FFNN(learning_rate, 0)
#     opt = optimizer
#     ann.add_hidden_layer(int(layer_1))
#     ann.add_hidden_layer(int(layer_2))
#     ann.add_hidden_layer(int(layer_3))
#     training_accuracy = 0
#     training_loss = 0
#     validation_loss = 0
#     validation_accuracy = 0
#     for x in range(int(epochs)):
#         for y in range(0 , len(data_x), batch_size):
#             training_loss, training_accuracy = ann.train(1, batch_size, data_x[y : y+batch_size], data_y[y: y+batch_size], x, opt)
#             validation_loss, validation_accuracy = ann.train(0, batch_size, val_x, val_y, x, opt)
#         wandb.log({"validation_accuracy" : validation_accuracy, "validation_loss" : validation_loss, "training_accuracy" : training_accuracy, "training_loss" : training_loss, "epochs": epochs})

def train(x_train, y_train):
    init_ = wandb.init(project="ann_from_scratch")
    # Access all hyperparameter values through wandb.config
    config = wandb.config
    wandb.run.name = "lr_" + str(config.learning_rate) + "_opt_" + str(config.optimizer) + "_epoch_" + str(config.epochs) + "_bs_" + str(config.batch_size) + "_act_" + str(config.activations)

    no_of_data_points = 6000
    train_x = x_train[:no_of_data_points]
    train_Y = y_train[:no_of_data_points]

    margin = int((0.1) * len(train_x))

    validation_x = train_x[:margin]
    validation_y = train_Y[:margin]

    training_x = train_x[margin:]
    training_y = train_Y[margin:]

    # logging(training_x, training_y, validation_x, validation_y, epoch=config.epochs, batch_size=config.batch_size, learning_rate=config.learning_rate, optimizer=config.optimizer, layer_1=config.layer_1, layer_2=config.layer_2, layer_3=config.layer_3)

    layer_1 = int(config.layer_1)
    layer_2 = int(config.layer_2)
    layer_3 = int(config.layer_3)
    optimizer = config.optimizer
    learning_rate = float(config.learning_rate)
    batch_size = config.batch_size
    if(optimizer == 'sgd'):
        batch_size = 1
    epochs = int(config.epochs)
    activation_function = config.activations
    data_x = training_x / 255
    data_y = training_y
    val_x = validation_x / 255
    val_y = validation_y
    
    # this value should be changeable
    loss_func = 'cross_ent'

    if(optimizer == "nag" or optimizer == 'nadam'):
        ann = FFNN(learning_rate, 1) # create FFNN with all these values
    else:
        ann = FFNN(learning_rate, 0)
    opt = optimizer
    ann.add_hidden_layer(int(layer_1))
    ann.add_hidden_layer(int(layer_2))
    ann.add_hidden_layer(int(layer_3))
    training_accuracy = 0
    training_loss = 0
    validation_loss = 0
    validation_accuracy = 0
    for x in range(int(epochs)):
        for y in range(0 , len(data_x), batch_size):
            training_loss, training_accuracy = ann.train(1, batch_size, data_x[y : y+batch_size], data_y[y: y+batch_size], x, opt, activation_function, loss_func)
            validation_loss, validation_accuracy = ann.train(0, batch_size, val_x, val_y, x, opt, activation_function, loss_func)
            wandb.log({"validation_accuracy" : validation_accuracy, "validation_loss" : validation_loss, "training_accuracy" : training_accuracy, "training_loss" : training_loss, "epochs": x})
train(train_x, train_Y)