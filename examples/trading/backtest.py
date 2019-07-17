import pandas_datareader.data as pdr
import yfinance as fix
import numpy as np
import tensorflow as tf
from examples.trading.utils import get_data_from_disc

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
    print(df_all.tail())
    print(df_all.shape)
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
    print('errors=',errors)
    for i in range(len(df_all)-1):
        print('i=',i)
        currBar = df_all[(i):]
        print(' next Bar=', df_all[(i+1):])
        print(' data[i]=', data[i])
        print(' data[i]=', closePrice[i])
        x_norm = tf.keras.utils.normalize(currBar, axis=1)
        y_train = keras.utils.to_categorical(y_train, size_output)
        x = np.array(closePrice.iloc[i: i , 1])
        y = np.array(closePrice.iloc[i + 1, 1])
        print('i=',i,'x=',x,'y=',y)
        prediction = model.predict(x)
        print('predict=',prediction)
        if (prediction == 1):
            buy += 1
        else:
            sell +=1
        if (prediction == y):
            success += 1
            balance += closePrice[i]
        else:
            fails += 1

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
