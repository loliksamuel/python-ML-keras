'''Trains a simple deep NN MLP (Multilayer perceptron) on the MNIST dataset. 28X28 images of digits

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# https://towardsdatascience.com/deep-learning-for-beginners-practical-guide-with-python-and-keras-d295bfca4487
# https://www.youtube.com/watch?v=aircAruvnKk
# print("creating deep learning to classify images to digit(0-9). MNIST images of 28×28 (=784 features) pixels are represented as an array of numbers  whose values range from [0, 255] (0=white,255=black) of type uint8")
# These MNIST images of 28×28 (=784 features) pixels are represented as an array of numbers  whose values range from [0, 255] (0=white,255=black) of type uint8.

def plot_image(df, title):
    plt.figure()
    plt.imshow(df[0])
    plt.colorbar()
    plt.gca().grid(False)
    plt.title(title)
    plt.show()

def plot_images(x,y, title):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i], cmap=plt.cm.binary)
        plt.xlabel(y[i])
    plt.show()

print('Loading train & test data')
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('\ntrain data')
print(x_train[0])
print(x_train[1])
print('shape', x_train.shape)
plot_image(x_train, 'picture example')
plot_images(x_train, y_train, 'picture examples')
print('\ntest data')
print(x_test[0])
print(x_test[1])

print('\nlabeled data')
print(y_train[0])
print(y_train[1])
print(y_train[2])
print(y_train[3])
print('shape', y_test.shape)

batch_size  = 128# we cannot pass the entire data into network at once , so we divide it to batches . number of samples that we will pass through the network at 1 time and use for each epoch. default is 32
epochs      = 1 #  iterations. on each, train all data, then evaluate, then adjust parameters (weights and biases)
#iterations  = 60000/128
num_classes = 10 # there are 10 classes (10 digits from 0 to 9)
num_hidden  = 512 # the bigger the number , the moe overfit, the lower the more underfit

#iterations  = 60000/128
print(x_train.shape[0], 'train samples')
print( x_test.shape[0], 'test samples')

print('\nNormalizing 0-255 (int)   to    0-1 (float)')
x_train = x_train.astype('float32')
x_test  =  x_test.astype('float32')
x_train /= 255
x_test  /= 255
#plot_image(x_test,'picture example')

print('\nConverting all 60000 images from matrix (28X28) to a vector (of size 784 features or neurons) cause only convolutional nn works with 2d images')
x_train = x_train.reshape(60000, 784)
x_test  =  x_test.reshape(10000, 784)



print('\nConvert class vectors to binary class matrices (for ex. convert digit 7 to bit array[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical( y_test, num_classes)
print('y_train[0]=', y_train[0])
print('y_test [0]=',  y_test[0])

print('\ncreate model...')
model = Sequential()# stack of layers
#tf.keras.layers.Flatten()
model.add(Dense  (num_hidden, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense  (num_hidden, activation='relu'))
model.add(Dropout(0.2))#regularization technic by removing some nodes
model.add(Dense  (num_classes, activation='softmax'))
# Prints a string summary of the  neural network.')
model.summary()
model.compile(loss      = 'categorical_crossentropy',# measure how accurate the model during training
              optimizer = RMSprop(),#this is how model is updated based on data and loss function
              metrics   = ['accuracy'])

print('\ntrain model...')
history = model.fit(  x_train
                    , y_train
                    , batch_size     = batch_size
                    , epochs         = epochs
                    , verbose        = 1
                    , validation_data= (x_test, y_test))

print('\nEvaluate the model with unseen data. pls validate that test accuracy =~ train accuracy and close to 1.0')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:'    , score[0], ' (is it close to 0?)')
print('Test accuracy:', score[1], ' (is it close to 1 and close to train accuracy?)')

print('\nPredict unseen data with 10 probabilities for 10 classes(choose the highest)')
predictions = model.predict(x_test)
print('labeled   as ', y_test[0]     , ' highest confidence for ' , np.argmax(y_test[0]))
print('predicted as ' ,predictions[0], ' highest confidence for ' , np.argmax(predictions[0]))


# plot

