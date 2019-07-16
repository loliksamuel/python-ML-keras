'''Trains a simple deep NN MLP (Multilayer perceptron) on the SP500

Gets to 90.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
1 seconds per epoch on a K520 GPU.

3 Techniques to Prevent Overfitting
1. Early Stopping: In this method, we track the loss on the validation set during the training phase and use it to determine when to stop training such that the model is accurate but not overfitting.
2. Image Augmentation: Artificially boosting the number of images in our training set by applying random image transformations to the existing images in the training set.
3. neoron Dropout: Removing a random selection of a fixed number of neurons in a neural network during training.


bug tracker
-----------------------------
priority | name
-----------------------------
1        | rubi, how accuracy so high??
2        | rubi, which normalize function to use?
'''

from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import utils as ut
import matplotlib.pyplot as plt
import random
#import pandas.io.data as web
import os
from sklearn.model_selection import train_test_split
# https://towardsdatascience.com/deep-learning-for-beginners-practical-guide-with-python-and-keras-d295bfca4487
# https://www.youtube.com/watch?v=aircAruvnKk




print('\nLoading  sp500 data')
print('\n======================================')
# Define date range
start_date, end_date='1970-01-03','2019-07-12'
dates=pd.date_range(start_date,end_date)
print("dates="  ,dates)
print("date[0]=",dates[0])

# Define stock symbols
symbols = []#'TSLA', 'GOOG', 'FB']  # SPY will be added in get_data()

# Get stock data
df_all = ut.get_data_from_disc('SP500')
print(df_all.tail())

# Slice and plot
#ut.plot_selected(df_all, [  'Close', 'sma200'], shouldNormalize=True)

# Slice and plot
ut.plot_selected(df_all, [ 'Close',  'sma200'],  shouldNormalize=False)
#plot_selected(df, ['Date','Close']                                    , start_date, end_date, shouldNormalize=False)
elements = df_all.size
shape=df_all.shape



print('\nsplit to train & test data')
print('\n======================================')
df_data = df_all.loc[:,   [ 'Open', 'High', 'Low', 'Close', 'sma10', 'sma20', 'sma50',  'sma200', 'range', 'range_sma']]
print('\ndata describe=\n',df_data.describe())
print('shape=',str(shape), " elements="+str(elements), ' rows=',str(shape[0]))
(x_train, x_test)  = train_test_split(df_data.values, test_size=0.33, shuffle=False)

print('\ntrain data', x_train.shape)
print(x_train[0])
print(x_train[1])

#plot_image(x_train, 'picture example')
#plot_images(x_train, y_train, 'picture examples')
print('\ntest data', x_test.shape)
print(x_test[0])
print(x_test[1])


print('\nLabeling')
print('\n======================================')
df_y = df_all['isUp']#np.random.randint(0,2,size=(shape[0], ))
#y = np.random.randint(0,2,size=(shape[0], ))
#print(y)
#df_y = pd.DataFrame()#, columns=list('is_up'))
#df_y['isUp'] = y
print(df_y)
(y_train, y_test)  = train_test_split(df_y.values, test_size=0.33, shuffle=False)
print(df_y.tail())
print('\nlabel describe\n',df_y.describe())
#(x_train, y_train)  = train_test_split(df.as_matrix(), test_size=0.33, shuffle=False)

print('\ntrain labels',y_train.shape)
print(y_train[0])
print(y_train[1])

print('\ntest labels', y_test.shape)
print(y_test[0])
print(y_test[1])

print(x_train.shape[0], 'train samples')
print( x_test.shape[0], 'test samples')
#df_GOOG = pd.read_csv('GOOG.csv', index_col="Date", parse_dates=True, usecols=["Date","Close","High","Low"], na_values=["nan"])

# df.GOOG.add(avg (df_GOOG))
# df.GOOG.add(atr (df_GOOG))
# df.GOOG.add(band(df_GOOG))
# double currAtr  = iATR(aSymbol, aTf,  aPeriod,  aShift);
# double currAtr  = iCustom(aSymbol, aTf, "AvgAtr3",aPeriod,   10,0,0,   20,0,0, 0,aShift);
# double avgAtr   = iCustom(aSymbol, aTf, "AvgAtr3",aPeriod,   10,0,0,   20,0,0, 1,aShift);
# double avgAtr2  = iCustom(aSymbol, aTf, "AvgAtr3",aPeriod,   10,0,0,   20,0,0, 2,aShift);
#currSpread
#currSwap
#economic
# (x_train, y_train), (x_test, y_test) = train_test_split(X, y_train, test_size=0.33



batch_size  = 128# we cannot pass the entire data into network at once , so we divide it to batches . number of samples that we will pass through the network at 1 time and use for each epoch. default is 32
epochs      = 25 #  iterations. on each, train all data, then evaluate, then adjust parameters (weights and biases)
#iterations  = 60000/128
num_input   = x_train.shape[1] # features
num_hidden  = 512 # If a model has more hidden units (a higher-dimensional representation space), and/or more layers, then the network can learn more complex representations. However, it makes the network more computationally expensive and may lead to overfit
num_classes = 2 # there are 3 classes (buy.sell. hold) or (green,red,hold)

#iterations  = 60000/128


print('\nClean data)')
print('\n======================================')
#dataset.isna().sum()
#dataset = dataset.dropna()

print('\nNormalize   to    0-1 (float)')
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test  = tf.keras.utils.normalize(x_test , axis=1)
#print('columns=', x_train.columns)
#print ('\ndf1=\n',x_train.loc[:, ['Open','High', 'Low', 'Close', 'range']])
#print ('\ndf1=\n',x_train.loc[:, ['sma10','sma20','sma50','sma200','range_sma']])
print(x_train[0])
print(x_train)
#print(x_train2[0])
#plot_image(x_test,'picture example')

print('\nRebalancing')


print('\nTransform data. Convert class vectors to binary class matrices (for ex. convert digit 7 to bit array[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical( y_test, num_classes)
print('y_train[0]=', y_train[0])
print('y_test [0]=',  y_test[0])

print('\ncreate model...')
print('\n======================================')
model = Sequential()# stack of layers
#model.add(tf.keras.layers.Flatten())
model.add(Dense  (num_hidden, activation='relu', input_shape=(num_input,)))
model.add(Dropout(0.2))
model.add(Dense  (num_hidden, activation='relu'))
model.add(Dropout(0.2))#regularization technic by removing some nodes
model.add(Dense  (num_classes, activation='softmax'))# last layer always has softmax(except for regession problems and  binary- 2 classes where sigmoid is enough)
# Prints a string summary of the  neural network.')
model.summary()
model.compile(loss      = 'categorical_crossentropy',# measure how accurate the model during training
              optimizer = RMSprop(),#this is how model is updated based on data and loss function
              metrics   = ['accuracy'])



layers = model.layers
#B_Output_Hidden = model.layers[0].get_weights()[1]
print ('model.weights=\n', model.get_weights())

print('\ntrain model for ',epochs,' epochs...')
print('\n======================================')
history = model.fit(  x_train
                    , y_train
                    , batch_size     = batch_size
                    , epochs         = epochs
                    , validation_data= (x_test, y_test)
                    , verbose        = 1
                  # , callbacks=[early_stop, PrintDot()]#Early stopping is a useful technique to prevent overfitting.
                      )


print('\nEvaluate the model with unseen data. pls validate that test accuracy =~ train accuracy and close to 1.0')
print('\n======================================')

print('\nplot_accuracy_loss_vs_time...')
history_dict = history.history
print(history_dict.keys())
ut.plot_stat_loss_vs_time     (history_dict)
ut.plot_stat_accuracy_vs_time (history_dict)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())
ut.plot_stat_train_vs_test(history)

score = model.evaluate(x_test, y_test, verbose=0)                                     # random                                |  calc label
print('Test loss:    ', score[0], ' (is it close to 0?)')                            #Test,train loss     : 0.6938 , 0.6933   |  0.47  0.5
print('Test accuracy:', score[1], ' (is it close to 1 and close to train accuracy?)')#Test,train accuracy : 0.5000 , 0.5000   |  0.69, 0.74

print('\nPredict unseen data with 10 probabilities for 10 classes(choose the highest)')
predictions = model.predict(x_test)
print('labeled   as ', y_test[0]     , ' highest confidence for ' , np.argmax(y_test[0]))
print('predicted as ' ,predictions[0], ' highest confidence for ' , np.argmax(predictions[0]))

filename='mnist_mlp.model'
print('\nSave model as ',filename)
model.save(filename)# 5.4 mb
newModel = tf.keras.models.load_model(filename)












