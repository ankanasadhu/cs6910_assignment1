from tkinter import Y
from Model import FFNN
from keras.datasets import fashion_mnist
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# from keras.datasets import mnist
import pandas as pd
import wandb

(train_x, train_Y), (test_x, test_Y) = fashion_mnist.load_data()
# (train_x, train_Y), (test_x, test_Y) = mnist.load_data()

hyp = {
    "learning_rate" : [0.01546597343, 0.060372463959, 0.03155889741],
    "is_nag" : [0, 1, 0],
    "batch_size" : [59, 153, 63],
    "activation" : ["logistic", "tanh", "tanh"],
    "optimizer" : ["adam", "nag", "momentum"],
    "epochs" : [20, 20, 18],
    "layer_1" : [128, 254, 128],
    "layer_2" : [64, 64, 64],
    "layer_3" : [16, 16, 16],
    "loss_func" : ['cross_ent', 'cross_ent', 'sq_err']
}

def train(x_train, y_train):

    #######
    init_ = wandb.init()
    ######

    no_of_data_points = 6000
    train_x_ = x_train[:no_of_data_points] / 255
    train_Y_ = y_train[:no_of_data_points]

    ########
    margin = int((0.1) * len(train_x_))
    validation_x = train_x_[:margin]
    validation_y = train_Y_[:margin]
    # print(train_x_.shape)
    # print(train_Y_.shape)
    train_x_ = train_x_[margin:]
    train_Y_ = train_Y_[margin:]
    ######

    ann = FFNN(hyp['learning_rate'][2], hyp['is_nag'][2])
    ann.add_hidden_layer(hyp['layer_1'][2])
    ann.add_hidden_layer(hyp['layer_2'][2])
    ann.add_hidden_layer(hyp['layer_3'][2])

    for x in range(int(hyp['epochs'][2])):
        for y in range(0 , len(train_x_), hyp['batch_size'][2]):
            if(y + hyp['batch_size'][2]) < len(train_x_):
                training_loss, training_accuracy = ann.train(1, hyp['batch_size'][2], train_x_[y : y+hyp['batch_size'][2]], train_Y_[y: y+hyp['batch_size'][2]], x, hyp['optimizer'][2], hyp['activation'][2], hyp['loss_func'][2])
                validation_loss, validation_accuracy = ann.train(0, hyp['batch_size'][2], validation_x, validation_y, x, hyp['optimizer'][2], hyp['activation'][2], hyp['loss_func'][2])
                wandb.log({"validation_accuracy" : validation_accuracy, "validation_loss" : validation_loss, "training_accuracy" : training_accuracy, "training_loss" : training_loss})
            else:
                break
        # print("Accuracy : " , training_accuracy)
        # print("Loss : ", training_loss)
        # print("val acc :", validation_accuracy)
        # print("val loss :", validation_loss)
    return ann

def save(ann, name_):
    file_to_store = open(name_ + ".pickle", "wb")
    pickle.dump(ann, file_to_store)
    file_to_store.close()

def test(ann, x_test, y_test):
    x_test = x_test/255
    i = 0
    count = 0
    predicted_y = []
    for img in x_test:
        true_y = np.zeros(10)
        true_y[y_test[i]] = 1
        activation = np.ndarray.flatten(img)   
        op = ann.forwardprop(activation, "logistic") 
        predicted_y.append(np.argmax(op))
        if(np.argmax(op) == y_test[i]):
            count += 1
        i += 1
    acc = count / len(x_test)
    predicted_y = np.array(predicted_y)
    return acc, predicted_y

def load_and_test(x_test, y_test):
    file_to_read = open("neural_net.pickle", "rb")
    loaded_object = pickle.load(file_to_read)
    file_to_read.close()
    acc, predicted_y = test(loaded_object, x_test, y_test)
    return acc , predicted_y

def create_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
                    index = ['T-shirt/top','Trouser','Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'], 
                    columns = ['T-shirt/top','Trouser','Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
    plt.figure(figsize=(10,10))
    sns.heatmap(cm_df, annot=True, fmt="d")
    plt.title('Confusion Matrix for Fashion MNIST dataset')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.savefig("confusion_matrix.png")


ffnn = train(train_x, train_Y)
# save(ffnn, "first_model_mnist")
print("Accuracy : ", test(ffnn, test_x, test_Y))



# acc, predictions = load_and_test(test_x, test_Y)
# create_confusion_matrix(test_Y, predictions)

