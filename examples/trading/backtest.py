import pandas_datareader.data as pdr
import yfinance as fix
import numpy as np
import tensorflow as tf
from examples.trading.utils import get_data_from_disc
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop

fix.pdr_override()


def back_test(model, symbol, start_date, end_date):
    """
    A simple back test for a given date period
    :param model     : the chosen strategy. Note to have already formed the model, and fitted with training data.
    :param symbol    : company ticker
    :param start_date: starting date :type  start_date : "YYYY-mm-dd"
    :param end_date  : ending date   :type  end_date  : "YYYY-mm-dd"

    :return: Percentage errors array that gives the errors for every test in the given date range
    """
    print('backtesting symbol ', symbol)
    print('==================================')
    df_all =  get_data_from_disc(symbol, 3600)
    df_x = df_all.loc[:,   [ 'Open', 'High', 'Low', 'Close', 'sma10', 'sma20', 'sma50',  'sma200', 'range', 'range_sma']]
    df_y = df_all['isUp']
    print(df_all.tail())
    print(df_all.shape)

    print('df_y=',df_y)
    size_output = 2
    y = keras.utils.to_categorical(df_y, size_output)
    print('y=',y)
    # data = pdr.get_data_yahoo(symbol, start_date, end_date)
    # closePrice = data["Close"]
    # print(closePrice)
    errors = []
    success = 0
    fails   = 0
    buy     = 0
    sell    = 0
    balance = 10000
    balanceStart = balance
    trades=[]
    print('errors=',errors)
    print('start trade...')
    for i in range(len(df_all)-1):
        print('i=',i)
        currBar = df_x[(i+0):(i+1)]
        nextBar = df_x[(i+1):(i+2)]
        nextBarIsUp = df_y[(i+1):(i+2)]
        print(' curr Bar=', currBar)
        print(' next Bar=', nextBar)
        print(' next Bar is Up? ', nextBarIsUp)

        # print(' data[i]=', data[i])
        # print(' data[i]=', closePrice[i])
        currBarNorm = tf.keras.utils.normalize(currBar, axis=1)


        print('i=',i,'x_norm=', currBarNorm)
        prediction = model.predict(currBarNorm)
        y_pred = np.argmax(prediction, axis=1)
        print('predict=',prediction, ' isUp?', y_pred)
        if  y_pred == 1 :
            buy += 1
        else:
            sell +=1
        profit = nextBar['range']
        print('profit=', profit)
        if  nextBarIsUp  and  y_pred == 1:
            success += 1

        else:
            fails += 1
        balance += profit
        trades.append(profit)
        print("longs={buy}, short = {sell}, fails = {fails} , success = {success}")
    # If you want to see the full error list then print the following statement
    print('errors=',errors)
    print("balance start:"+balanceStart)
    print("balance end  :"+balance)
    print("profit       :"+str(balance-balanceStart)+' points')
    print("trades       :"+str(buy+sell))
    print("trade long   :"+str(buy))
    print("trade short  :"+str(sell))


symbol='^GSPC'# ^GSPC = SP500

epochs=50
size_hidden=512

filename = 'mlpt_'+symbol+'_'+str(epochs)+'_'+str(size_hidden)+'.model'
model    = tf.keras.models.load_model(filename)
print('\nBacktesting')
print('\n======================================')
back_test(model, symbol, start_date='1970-01-03', end_date='2019-05-05')
