'''Trains a simple deep NN MLP (Multilayer perceptron) on the MNIST dataset. 28X28 images of digits

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.


3 Techniques to Prevent Overfitting
1. Early Stopping: In this method, we track the loss on the validation set during the training phase and use it to determine when to stop training such that the model is accurate but not overfitting.
2. Image Augmentation: Artificially boosting the number of images in our training set by applying random image transformations to the existing images in the training set.
3. neoron Dropout: Removing a random selection of a fixed number of neurons in a neural network during training.

'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import random
#import pandas.io.data as web
import os
from sklearn.model_selection import train_test_split
# https://towardsdatascience.com/deep-learning-for-beginners-practical-guide-with-python-and-keras-d295bfca4487
# https://www.youtube.com/watch?v=aircAruvnKk
# print("creating deep learning to classify images to digit(0-9). MNIST images of 28×28 (=784 features) pixels are represented as an array of numbers  whose values range from [0, 255] (0=white,255=black) of type uint8")
# These MNIST images of 28×28 (=784 features) pixels are represented as an array of numbers  whose values range from [0, 255] (0=white,255=black) of type uint8.



def kpi_returns(prices):
    return ((prices-prices.shift(-1))/prices)[:-1]



def kpi_sharpeRatio():

    risk_free_rate = 2.25 # 10 year US-treasury rate (annual) or 0
    sharpe = 2
    #  ((mean_daily_returns[stocks[0]] * 100 * 252) -  risk_free_rate ) / (std[stocks[0]] * 100 * np.sqrt(252))
    return sharpe

def kpi_commulativeReturn():
    return 2


def kpi_risk(df):
    return df.std()


def kpi_sharpeRatio():
    return 2



def loss_log():
    return 2

def loss_mse():
    return 2


def loss_gdc():
    return 2

def activation_sigmoid():
    return 2


def activation_softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div

def activation_reLU():
    return 2

def plot_selected(df, columns, shouldNormalize = True):
    """Plot the desired columns over index values in the given range."""
    #df = df[columns][start_index:end_index]
    #df = df.loc[start_index:end_index, columns]
    df = df.loc[:, columns]
    ylabel="Price"
    normal = "un normalized"
    if shouldNormalize:
        df = normalize(df.loc[:,['Close',   'sma200']])
        ylabel = "%"
        normal = "normalized"
    print('df.shape in plot=',df.shape)
    plot_data(df, title='stock price ('+normal+')', ylabel=ylabel)




def plot_data(df, title="normalized Stock prices", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    plt.show()

def plot_image(df, title):
    plt.figure()
    plt.imshow(df[0])#, cmap=plt.cm.binary)
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

def plot_stat_loss_vs_time(history_dict) :
    acc      = history_dict['acc']
    val_acc  = history_dict['val_acc']
    loss     = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss   , 'bo', label='train loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='test loss')
    plt.title('train & test loss over time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_stat_accuracy_vs_time(history_dict) :
    acc      = history_dict['acc']
    val_acc  = history_dict['val_acc']
    loss     = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc    , 'bo', label='train acc')
    plt.plot(epochs, val_acc, 'b' , label='test acc')
    plt.title('train & test accuracy over time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def plot_stat_train_vs_test(history):
    hist = history.history
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.plot(hist['loss'])#train loss
    plt.plot(hist['val_loss'])#validation loss
    plt.title    ('model loss')
    plt.legend   (['train Error', 'test Error'], loc='upper right')
    plt.show()

# normalize to first row
def normalize(df):
    return df/df.iloc[0,:]


def normalize2(x):
    train_stats = x_train.describe()
    return (x - train_stats['mean']) / train_stats['std']


def symbol_to_path(symbol, base_dir=""):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data_from_disc_join(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'GOOG' not in symbols:  # add GOOG for reference, if absent
        symbols.insert(0, 'GOOG')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                              parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])

        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        print(df_temp.head())
        df = df.join(df_temp)
        if symbol == 'GOOG':  # drop dates GOOG did not trade
            df = df.dropna(subset=["GOOG"])

    return df

def get_data_from_disc(symbol):
    """Read stock data (adjusted close) for given symbols from CSV files.
    https://finance.yahoo.com/quote/%5EGSPC/history?period1=-630986400&period2=1563138000&interval=1d&filter=history&frequency=1d
    """

    df1 = pd.read_csv(  symbol_to_path(symbol)
                          , index_col  = 'Date'
                          , parse_dates= True
                          , usecols    = ['Date', 'Close', 'Open', 'High', 'Low']
                          , na_values  = ['nan'])



    df1['sma10'] = df1['Close'].rolling(window=10).mean()
    df1['sma20'] = df1['Close'].rolling(window=20).mean()
    df1['sma50'] = df1['Close'].rolling(window=50).mean()
    df1['sma200'] = df1['Close'].rolling(window=200).mean()

    df1 = df1[-(df1.shape[0]-200):]  # skip 1st 200 rows due to NAN in sma, range
    print ('\ndf1=\n',df1)
    print ('\nsma_10=\n',df1['sma10'] )
    print ('\nsma_20=\n',df1['sma20'] )

    df1['range'] = df1['Close']-df1['Open']
    print ('\nrange=\n',df1['range'])
    df1['range_sma'] = df1['sma10'] - df1['sma20']
    #df1['isUp'] = 0
    print(df1)
    #df1['isUp']  = np.random.randint(2, size=df1.shape[0])
    # if df1['range'] > 0.0:
    #     df1['isUp'] = 1
    # else:
    #     df1['isUp'] = 0
    df1.loc[df1.range >  0.0, 'isUp'] = 1
    df1.loc[df1.range <= 0.0, 'isUp'] = 0

#direction = (close > close.shift()).astype(int)
    #target = direction.shift(-1).fillna(0).astype(int)
    #target.name = 'target'
    #sma10 = sma10.rename(columns={symbol: symbol+'sma10'})
    #sma20 = sma20.rename(columns={symbol: symbol+'sma20'})
    #df1 = df1.rename(columns={'Close': symbol+'Close'})


    print('columns=', df1.columns)
    print ('\ndf1=\n',df1)
    return df1


def get_data_from_web(symbol):
    start, end = '1950-01-03','2019-07-12'#'2007-05-02', '2016-04-11'
    data = web.DataReader(symbol, 'yahoo', start, end)
    data=pd.DataFrame(data)
    prices=data['Adj Close']
    prices=prices.astype(float)
    return prices


def get_state(parameters, t, window_size = 20):
    outside = []
    d = t - window_size + 1
    for parameter in parameters:
        block = (
            parameter[d : t + 1]
            if d >= 0
            else -d * [parameter[0]] + parameter[0 : t + 1]
        )
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        for i in range(1, window_size, 1):
            res.append(block[i] - block[0])
        outside.append(res)
    return np.array(outside).reshape((1, -1))


def kpi_sharpeRatio():
    return 2









print('\nLoading  data')
print('\n======================================')
# Define date range
start_date, end_date='1950-01-03','2019-07-12'
dates=pd.date_range(start_date,end_date)
print("dates="  ,dates)
print("date[0]=",dates[0])

# Define stock symbols
symbols = []#'TSLA', 'GOOG', 'FB']  # SPY will be added in get_data()

# Get stock data
df_all = get_data_from_disc('SP500')
print(df_all.tail())

# Slice and plot
plot_selected(df_all, [  'Close', 'sma200'], shouldNormalize=True)

# Slice and plot
plot_selected(df_all, [ 'Close',  'sma200'],  shouldNormalize=False)
#plot_selected(df, ['Date','Close']                                    , start_date, end_date, shouldNormalize=False)
elements = df_all.size
shape=df_all.shape



print('\nsplit to train & test data')
print('\n======================================')
df_data = df_all.loc[:,   [ 'Open', 'High', 'Low', 'Close', 'sma10', 'sma20', 'sma50',  'sma200', 'range', 'range_sma']]
print('\ndata stats=\n',df_data.describe())
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
epochs      = 250 #  iterations. on each, train all data, then evaluate, then adjust parameters (weights and biases)
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
print(x_train[0])
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



print('\ntrain model...')
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
plot_stat_loss_vs_time     (history_dict)
plot_stat_accuracy_vs_time (history_dict)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())
plot_stat_train_vs_test(history)

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












