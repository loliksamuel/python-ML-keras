import pandas_datareader.data as pdr
import yfinance as fix
import numpy as np
import tensorflow as tf
import pandas as pd
from examples.trading.utils import get_data_from_disc, plot_data, plot_list, plot_live, plot_barchart
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
fix.pdr_override()


def back_test(filename, symbol, start_date, end_date):
    """
    A simple back test for a given date period
    :param model     : the chosen strategy. Note to have already formed the model, and fitted with training data.
    :param symbol    : company ticker
    :param start_date: starting date :type  start_date : "YYYY-mm-dd"
    :param end_date  : ending date   :type  end_date  : "YYYY-mm-dd"

    :return: Percentage errors array that gives the errors for every test in the given date range
    """
    skipRows = 17400#3600 6600

    model    = tf.keras.models.load_model('models/',filename)

    print('loading data of symbol ', symbol)
    print('==================================')
    df_all =  get_data_from_disc(symbol,skipRows )
    df_x = df_all.loc[:,   ['sma10', 'sma20', 'sma50',  'sma200',  'sma400', 'range_sma', 'range_sma1', 'range_sma2', 'range_sma3',  'range_sma4', 'bb_hi10', 'bb_lo10', 'bb_hi20', 'bb_lo20', 'bb_hi50', 'bb_lo50', 'bb_hi200', 'bb_lo200', 'rel_bol_hi10', 'rel_bol_lo10', 'rel_bol_hi20', 'rel_bol_lo20', 'rel_bol_hi50', 'rel_bol_lo50', 'rel_bol_hi200', 'rel_bol_lo200', 'rsi10', 'rsi20', 'rsi50', 'rsi200', 'stoc10', 'stoc20', 'stoc50', 'stoc200']]
    df_oc = df_all.loc[:,   [ 'range', 'Open' , 'Close' ]]
    df_y_true = df_all['isUp']
    print(df_all.tail())
    print(df_all.shape)
    print('df_y=',df_y_true.shape, '\n', df_y_true)

    df_x = tf.keras.utils.normalize(df_x, axis=1)
    print('predicting... ')
    df_y_pred = model.predict(df_x)

    size_output = 2
    lenxx = len(df_y_true)
    y = keras.utils.to_categorical(df_y_true, size_output)
    print('len=',lenxx,' y=',y)
    # data = pdr.get_data_yahoo(symbol, start_date, end_date)
    # closePrice = data["Close"]
    # print(closePrice)

    win_long   = 0
    win_short  = 0
    lose_long  = 0
    lose_shrt  = 0
    pointUsdRatio = 1
    initialDeposit = 10000
    profitCurr   = 0
    listTrades   =[]
    listLongs    =[]
    listShorts   =[]
    listWinners  =[]
    listLosers   =[]
    plt.clf()
    title="commulative profit over time"
    xlabel="trades"
    ylabel="points"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    print('start trading...')
    for i in range(lenxx-1):
        currBar     = df_oc[(i+0):(i+1)]
        nextBar     = df_oc[(i+1):(i+2)]
        nextBarIsUp = df_y_true[(i+1):(i+2)]
        print('\n#',i,'out of ',lenxx,'. curr Bar =', currBar)
        bar_range = nextBar['range'].iloc[0]
        open      = nextBar ['Open'].iloc[0]
        close     = nextBar['Close'].iloc[0]
        print(' next Bar range=', bar_range)

        #print(' curr Bari=', currBar.iloc)
        nextBarUp = nextBarIsUp.values[0]
        #print(' next Bar is Up? ', nextBarIsUp)
        print(' next Bar is Up? ', nextBarIsUp.values[0])

        # print(' data[i]=', data[i])
        # print(' data[i]=', closePrice[i])


        y_pred_all = np.argmax(df_y_pred, axis=1)

        # print('predict=',prediction, ' isUp?', y_pred, ' range=',profit, ' y_pred=',y_pred)
        y_pred_next = y_pred_all[(i+1):(i+2)]
        if  y_pred_next == 1 :# green bar prediction
            profitCurr = bar_range
            if  nextBarUp == 1 :
                listWinners.append(profitCurr)
                win_long    += 1
            else:
                listLosers.append(profitCurr)
                lose_long   += 1
            listLongs.append(profitCurr)

            print(' buy @' , str(round(open,2)), ' exit @',str(round(close,2)) , ' profit = ', profitCurr   )
        else:# red bar prediction
            profitCurr = -bar_range
            if  nextBarUp == 1  :
                listLosers.append(profitCurr)
                lose_shrt    += 1
            else:
                listWinners.append(profitCurr)
                win_short    += 1
            listShorts.append(profitCurr)

            print(' sell @' , str(round(open,2)), ' exit @',str(round(close,2)), ' profit = ', profitCurr   )


        #balanceTotal += profit
        listTrades.append(profitCurr)
        #plot_live(np.cumsum(listTrades, dtype=float) , title="commulative profit over time", xlabel="trades",  ylabel="points")

        cumsum = np.cumsum(listTrades, dtype=float)

        plt.plot(  i,  cumsum[i], '.b' )# - is line , b is blue
        plt.draw()
        plt.pause(0.01)


        print('longs=',str(len(listLongs)), ' short=',str(len(listShorts)), ' gain_all=',str(len(listWinners)) , ' loss_all=',str(len(listLosers)), ' profitCurr=', str(round(profitCurr,2)), ' profitTotal=', round(sum(listTrades),2), ' profitShorts=', round(sum(listShorts,2)), ' profitLongs=', round(sum(listLongs),2))

    # If you want to see the full error list then print the following statement
    longs  = len(listLongs)
    shorts = len(listShorts)
    totalTrades  = len(listTrades)
    totalProfit  = sum(listTrades)
    profitLongs  = sum(listLongs)
    profitShorts = sum(listShorts)
    profitWinner = sum(listWinners)
    profitLosers = sum(listLosers)
    print('\nsummary\n==================' )
    print('symbol=',symbol )
    print('period=', lenxx ,' bars' )
    print('strategy=',filename )
    print("initial  deposit : "+str(initialDeposit))
    print("Expected payoff  : (all/long/short/win/lose)   : ",round(totalProfit/totalTrades,2) , ' / ', round(profitLongs/longs,2) , ' / ', round(profitShorts/shorts,2) , ' / ', round(profitWinner/len(listWinners),2) , ' / ', round(profitLosers/len(listLosers),2) , ' points')
    print("absolute drawdown: \n")

    print("total net profit : ",str(round(totalProfit,2))+' $, ' , str(round(totalProfit/initialDeposit*100,2))+'%')
    print("  profits from longs  : " + str(round( profitLongs ,2))+' $ , ',str(round(profitLongs/ totalProfit*100,2)),'% of total')
    print("  profits from shorts : " + str(round( profitShorts,2))+' $ , ',str(round(profitShorts/totalProfit*100,2)),'% of total')
    print("total positions    : " ,totalTrades     , '# , ', str(round((win_long+win_short)/totalTrades *100,2)), '% won ' )
    print("  longs positions  : " + str(longs )    , '# , ', str(round(win_long/longs *100,2)) , '% won,  largest=',str(round(max( listLongs),2))  , '$, smallest=',str(round(min(listLongs) ,2)))
    print("  shorts positions : " + str(shorts)    , '# , ', str(round(win_short/shorts*100,2)), '% won,  largest=',str(round(max(listShorts),2))  , '$, smallest=',str(round(min(listShorts),2)))
    print("  winner positions : ", len(listWinners), '# , ', round(len(listWinners)/totalTrades*100,2), '% of total ,' ,round(sum(listWinners),2),'$')
    print("  loser  positions : " ,len( listLosers), '# , ', round(len(listLosers) /totalTrades*100,2), '% of total ,' ,round(sum(listLosers ),2),'$')
    plt.clf()
    plot_barchart(listTrades, title="BT-trade profit over time", xlabel="trades",  ylabel="points")

    listTrades.insert(1,initialDeposit)
    plt.clf()
    plot_list(np.cumsum(listTrades, dtype=float) , title="BT-commulative profit over time", xlabel="trades",  ylabel="points")

    plt.clf()
    title="BT-profit per year"
    for i in range(1, 13):
        plt.subplot(3, 4, i)
        plot_list(np.cumsum(listTrades[i*1000+890:i*1000+1890], dtype=float) , title=title, xlabel="trades yr #"+str(i),  ylabel="points", dosave=0)
    # plot_list(np.cumsum(listTrades.index(1000,2000), dtype=float) , title="commulative profit over time 2nd year", xlabel="trades",  ylabel="points")
    # plot_list(np.cumsum(listTrades.index(4000,5000), dtype=float) , title="commulative profit over time 3rd year", xlabel="trades",  ylabel="points")
    # plot_list(np.cumsum(listTrades.index(7000,8000), dtype=float) , title="commulative profit over time 4st year", xlabel="trades",  ylabel="points")
    # plot_list(np.cumsum(listTrades.index(1000,-1), dtype=float) , title="commulative profit over time last  year", xlabel="trades",  ylabel="points")
    # plot_list(np.cumsum(listTrades, dtype=float) , title="commulative profit over time", xlabel="trades",  ylabel="points")
    plt.savefig('plots/bt/'+title+'.png')
    #print (listTrades)



print('\nBacktesting')
print('\n=========================================')
symbol='^GSPC'# ^GSPC = SP500
epochs=50
size_hidden=512
filename = 'mlpt_'+symbol+'_'+str(epochs)+'_'+str(size_hidden)+'.model'

back_test(filename, symbol, start_date='1970-01-03', end_date='2019-05-05')
