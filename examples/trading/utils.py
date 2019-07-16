

from __future__ import absolute_import, division, print_function, unicode_literals

import math
# import pandas.io.data as web
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

def plot_stat_loss_vs_accuracy(history_dict) :
    acc_train  = history_dict['acc']
    acc_test   = history_dict['val_acc']
    loss_train = history_dict['loss']
    loss_test  = history_dict['val_loss']

    epochs = range(1, len(acc_train) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss_train   , 'b', color='blue',label='train loss')
    plt.plot(epochs, loss_test    , 'b', color='green',label='test_loss')
    # b is for "solid blue line"
    plt.plot(epochs, acc_train, 'b', color='red'   , label='train accuracy')
    plt.plot(epochs, acc_test , 'b', color='orange', label='test  accuracy')
    plt.title('train & test loss & accuracy over time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss& accuracy')
    plt.legend()
    plt.show()

def plot_stat_loss_vs_time(history_dict) :
    acc_train  = history_dict['acc']
    acc_test   = history_dict['val_acc']
    loss_train = history_dict['loss']
    loss_test  = history_dict['val_loss']
    epochs = range(1, len(acc_train) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss_train   , 'bo', label='train loss')
    # b is for "solid blue line"
    plt.plot(epochs, loss_test, 'b', label='test loss')
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
    train_stats = x.describe()
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

    df1 = df1[-(df1.shape[0]-3600):]  # skip 1st x rows, x years due to NAN in sma, range
    print ('\ndf1=\n',df1.tail())
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
    print ('\ndf1=\n',df1.loc[:, ['Open','High', 'Low', 'Close', 'range', 'isUp']])
    print ('\ndf1=\n',df1.loc[:, ['sma10','sma20','sma50','sma200','range_sma']])
    return df1


def get_data_from_web(symbol):
    start, end = '1970-01-03','2019-07-12'#'2007-05-02', '2016-04-11'
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


def plot_histogram(x, bins, title, xlabel, ylabel):
    plt.hist(x, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()









