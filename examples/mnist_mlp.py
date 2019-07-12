'''Trains a simple deep NN MLP (Multilayer perceptron) on the MNIST dataset. 28X28 images of digits

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

# import matplotlib.pyplot as plt

# https://towardsdatascience.com/deep-learning-for-beginners-practical-guide-with-python-and-keras-d295bfca4487
# https://www.youtube.com/watch?v=aircAruvnKk
# print("creating deep learning to classify images to digit(0-9). MNIST images of 28×28 (=784 features) pixels are represented as an array of numbers  whose values range from [0, 255] (0=white,255=black) of type uint8")
# These MNIST images of 28×28 (=784 features) pixels are represented as an array of numbers  whose values range from [0, 255] (0=white,255=black) of type uint8.


print('Loading data...split between train and test sets')
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('train data')
print(x_train[0])
print(x_train[1])

print('test data')
print(x_test[0])
print(x_test[1])

print('y data')
print(y_test[0])
print(y_test[1])
print(y_test[2])
print(y_test[3])

batch_size  = 128
num_classes =  10
epochs      =  3
#iterations  = 60000/128

# converting all 60000 images from matrix (28X28) to a vector (of size 784 features or neurons)
x_train = x_train.reshape(60000, 784)
x_test  =  x_test.reshape(10000, 784)

# normalizing 0-255 (int)   to    0-1 (float)
x_train = x_train.astype('float32')
x_test  =  x_test.astype('float32')
x_train /= 255
x_test  /= 255
print(x_train.shape[0], 'train samples')
print( x_test.shape[0], 'test samples')

print('convert class vectors to binary class matrices (for ex. convert digit 7 to bit array[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical( y_test, num_classes)

print('Build model...')
model = Sequential()
model.add(Dense  (512        , activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense  (512        , activation='relu'))
model.add(Dropout(0.2))#regularization technic by removing some nodes
model.add(Dense  (num_classes, activation='softmax'))

# Prints a string summary of the  neural network.
model.summary()

model.compile(loss     ='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics  =['accuracy'])

print('Building model...')
history = model.fit(  x_train
                    , y_train
                    , batch_size     = batch_size
                    , epochs         = epochs
                    , verbose        = 1
                    , validation_data= (x_test, y_test))

# evaluate the model with unseen data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:'    , score[0])
print('Test accuracy:', score[1])

# predict unseen data
# x_test = [0,0,0,0,1,0,0,0,0,0]
# predictions = model.predict(x_test)
# print(predictions[11])


# plot
