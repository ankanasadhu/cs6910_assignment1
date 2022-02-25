from turtle import end_fill
from matplotlib import pyplot
from keras.datasets import fashion_mnist
import numpy as np

(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
label= ["T-shirt/top","Trouser","Pullover",	"Dress","Coat",	"Sandal" ,"Shirt","Sneaker","Bag","Ankle boot"]
c=[0]*10
count=0
# summarize loaded dataset

i=0
while count<10 :
	if c[train_y[i]]==0 :
		pyplot.subplot(3,4, (1 + count)) 
		pyplot.title(label[train_y[i]])
		pyplot.tight_layout()
		pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
		c[train_y[i]]+=1
		count+=1
	i+=1

# show the figure
pyplot.show()