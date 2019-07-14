'''Trains a simple deep NN linear regression (Multilayer perceptron) on the MNIST dataset. 28X28 images of digits

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

from keras.layers import Dense
from keras.models import Model, Sequential
from keras import initializers

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



def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div

def loss_log():
    return 2

def loss_mse():
    return 2


def loss_gdc():
    return 2

def activation_sigmoid():
    return 2


def plot_selected(df, columns, start_index, end_index):
    """Plot the desired columns over index values in the given range."""
    # TODO: Your code here
    # Note: DO NOT modify anything else!
    #df = df[columns][start_index:end_index]
    df.ix[start_index:end_index, columns]
    df = normalize(df)
    plot_data(df)

def plot_data(df, title="normalized Stock prices"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
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
    plt.plot(epochs, loss   , 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss over time')
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

    plt.plot(epochs, acc    , 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b' , label='Validation acc')
    plt.title('Training and validation accuracy over time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def plot_stat_train_vs_test(history):
    hist = history.history
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.title    ('model loss')
    plt.legend   (['train Error', 'test Error'], loc='upper right')
    plt.show()


# normalize to first row
def normalize(df):
    return df/df.ix[0,:]


def normalize(x):
    train_stats = x_train.describe()
    return (x - train_stats['mean']) / train_stats['std']


def symbol_to_path(symbol, base_dir=""):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data_from_disc(symbols, dates):
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

def get_data_from_web(symbol):
    start, end = '2007-05-02', '2016-04-11'
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

## Set the mean, standard deviation, and size of the dataset, respectively
mu, sigma, size = 0, 4, 100

## Set the slope (m) and y-intercept (b), respectively
m, b = 2, 100

## Create a uniformally distributed set of X values between 0 and 10 and store in pandas dataframe
x = np.random.uniform(0,10, size)
df = pd.DataFrame({'x':x})

## Find the "perfect" y value corresponding to each x value given
df['y_perfect'] = df['x'].apply(lambda x: m*x+b)


## Create some noise and add it to each "perfect" y value to create a realistic y dataset
df['noise'] = np.random.normal(mu, sigma, size=(size,))
df['y'] = df['y_perfect']+df['noise']

## Plot our noisy dataset with a standard linear regression
## (note seaborn, the plotting library, does the linear regression by default)
ax1 = sns.regplot(x='x', y='y', data=df)

print ('\ndata = ',df)


## Create our model with a single dense layer, with a linear activation function and glorot (Xavier) input normalization
print('\ncreate model...')
#model = Sequential([Dense(1, activation='linear', input_shape=(1,), kernel_initializer='glorot_uniform')])

model = Sequential()# stack of layers
model.add(Dense  (1, activation='linear', input_shape=(1,), kernel_initializer='glorot_uniform'))
model.summary()
## Compile our model using the method of least squares (mse) loss function
## and a stochastic gradient descent (sgd) optimizer
model.compile(  loss     ='mse'
              , optimizer='sgd') ## To try our model with an Adam optimizer simple replace 'sgd' with 'Adam'

## Set our learning rate to 0.01 and print it
#model.optimizer.lr.set_value(.001)
#print ('optimizer=',model.optimizer.lr.get_value())

## Fit our model to the noisy data we create above. Notes:
## The validation split parameter reserves 20% of our data for validation (ie 80% will be used for training)
## The callback parameter is where we tell our model to use the callback function created above
## I don't really know if using a batch size of 1 makes sense
history = model.fit(    x=df['x']
                      , y=df['y']
                      , validation_split= 0.2
                      , batch_size      = 1
                      , epochs          = 100
                    #  , callbacks=[print_save_weights]
                        )

## As the model is fitting the data you can watch below and see how our m and b parameters are improving

## Save and print our final weights
predicted_m = model.get_weights()[0][0][0]
predicted_b = model.get_weights()[1][0]
print ("\nm=%.2f b=%.2f\n" % (predicted_m, predicted_b))#m=2.45 b=99.48

plot_stat_train_vs_test(history)

# Plot our model's prediction over the data and real line
## Create our predicted y's based on the model
df['y_predicted'] = df['x'].apply(lambda x: predicted_m*x + predicted_b)
## Plot the original data with a standard linear regression
ax1 = sns.regplot(x='x', y='y', data=df, label='real')

## Plot our predicted line based on our Keras model's slope and y-intercept
ax2 = sns.regplot(x='x', y='y_predicted', data=df, scatter=False, label='predicted')
ax2.legend(loc="upper left")





