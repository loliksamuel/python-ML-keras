'''Trains and evaluate a simple MLP(Multilayer perceptron)
on the Reuters newswire topic classification task.
'''
from __future__ import print_function

import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

max_words   = 1000

print('Loading data...split between train and test sets')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                    test_split=0.2)
print('train data')
print(x_train[0])
print(x_train[1])

print('test data')
print(x_test[0])
print(x_test[1])

print('y data')
print(y_test[0])
print(y_test[1])


batch_size  = 32
num_classes = np.max(y_train) + 1
epochs      = 5

print(len(x_train), 'train sequences')
print(len(x_test ), 'test sequences')


print(num_classes, 'classes')

print('\nVectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)
x_train   = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test    = tokenizer.sequences_to_matrix(x_test , mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:' , x_test.shape)
print('x_test:' , x_test[0])

print('\nConvert class vector to binary class matrix  (for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test , num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:' , y_test.shape)

print('\nBuild model...')
model = Sequential()
model.add(Dense     (512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout   (0.5))
model.add(Dense     (num_classes))
model.add(Activation('softmax'))

model.compile(loss     ='categorical_crossentropy',
              optimizer='adam',
              metrics  =['accuracy'])

print('\nTrain model...')
history = model.fit(  x_train
                    , y_train
                    , batch_size      = batch_size
                    , epochs          = epochs
                    , verbose         = 1
                    , validation_split= 0.1)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

print('Test score:'   , score[0])
print('Test accuracy:', score[1])
