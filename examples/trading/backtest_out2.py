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


def back_test(filename, symbol, skipRows, names_input,names_output, start_date, end_date):
    """
    A simple back test for a given date period
    :param model     : the chosen strategy. Note to have already formed the model, and fitted with training data.
    :param symbol    : company ticker
    :param start_date: starting date :type  start_date : "YYYY-mm-dd"
    :param end_date  : ending date   :type  end_date  : "YYYY-mm-dd"

    :return: Percentage errors array that gives the errors for every test in the given date range
    """

    print('\nLoading model')
    model    = tf.keras.models.load_model('files/output/'+filename)

    print('\nLoading data of symbol ', symbol)
    df_all =  get_data_from_disc(symbol,skipRows )

    print('\nExtrating x')
    df_x = df_all.loc[:,   names_input]
    df_oc = df_all.loc[:,   [ 'range', 'Open' , 'Close' ]]

    print('\nExtrating y')
    df_y_observed = df_all['isUp']
    print(df_all.tail())
    print(df_all.shape)
    print('df_y=',df_y_observed.shape, '\n', df_y_observed)

    print('\nNormalizing... ')
    df_x = tf.keras.utils.normalize(df_x, axis=1)

    print('\nPredicting... ')
    df_y_pred = model.predict(df_x)

    size_output  = len(names_output)
    lenxx = len(df_y_observed)
    y = keras.utils.to_categorical(df_y_observed, size_output)
    print('len=',lenxx,' y=',y)
    # data = pdr.get_data_yahoo(symbol, start_date, end_date)
    # closePrice = data["Close"]
    # print(closePrice)
    start = 0 #use start=9308 to start from index where validation set start
    win_long   = 0
    win_short  = 0
    lose_long  = 0
    lose_shrt  = 0
    pointUsdRatio = 1
    initialDeposit = 10000
    pointsCurr   = 0
    percentCurr  = 0
    listTradesPercent = []
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
    '''
    # 0 out of  13894 . 
    Date        range     Open       Close
    1964-05-01  0.709999  79.459999  80.169998
     next Bar range= 0.3000030000000038
     next Bar is Up?  1.0
     buy @ 80.17  exit @ 80.47  profit =  0.3000030000000038
    longs= 1  short= 0  gain_all= 1  loss_all= 0  profitCurr= 0.3  profitTotal= 0.3  profitShorts= 2  profitLongs= 0.3
    
    symbol= ^GSPC
    period= 13894  bars
    strategy= mlpt_^GSPC_100_512.model
    initial  deposit : 10,000
    Expected payoff  : (all/long/short/win/lose)   :  2.66  /  2.2  /  3.49  /  5.42  /  -3.82  points
    absolute drawdown: 100$
    total net profit :  36,984.91 $,  369.85%
      profits from longs  : 19660.32 $ ,  53.16 % of total
      profits from shorts : 17324.59 $ ,  46.84 % of total
    total positions    :  13893 # ,  70.17 % won 
      longs positions  : 8927 # ,  67.69 % won,  largest= 104.58 $, smallest= -86.76
      shorts positions : 4966 # ,  74.63 % won,  largest= 104.01 $, smallest= -80.36
      winner positions :  9749 # ,  70.17 % of total , 52815.58 $
      loser  positions :  4144 # ,  29.83 % of total , -15830.67 $
    
    summary
    ==================
    symbol= DJI
    period= 8385  bars
    strategy= mlpt_DJI_50_512.model
    initial  deposit : 10000
    Expected payoff  : (all/long/short/win/lose)   :  -2.85  /  0.0  /  -2.85  /  68.35  /  -65.02  points
    absolute drawdown: -23,907
    
    total net profit :  -23,907 $,  -239.08%
      profits from longs  : 0 $ ,  -0.0 % of total
      profits from shorts : -23907.65 $ ,  100.0 % of total
    total positions    :  8384 # ,  46.61 % won 
    '''
    for i in range(start,lenxx-1):#0 to 13894   #for index, row in df_oc.iterrows():
        currBar      = df_oc[(i+0):(i+1)]
        #nextBar     = df_oc[(i+1):(i+2)]
        currBarIsUp  = df_y_observed[(i+0):(i+1)]
        print('\n#',i,'out of ',lenxx,'. curr Bar =', currBar)
        bar_range = currBar['range'].iloc[0]  #  bar_range = row['range']
        open      = currBar[ 'Open'].iloc[0]  #  bar_range = row['Open']
        close     = currBar['Close'].iloc[0]  #  bar_range = row['Close']
        print(' curr Bar range=', bar_range)

        #print(' curr Bari=', currBar.iloc)
        currBarUp = currBarIsUp.values[0]
        #print(' next Bar is Up? ', nextBarIsUp)
        print(' curr Bar is Up? ', currBarIsUp.values[0])

        # print(' data[i]=', data[i])
        # print(' data[i]=', closePrice[i])


        y_pred_all = np.argmax(df_y_pred, axis=1)

        # print('predict=',prediction, ' isUp?', y_pred, ' range=',profit, ' y_pred=',y_pred)
        y_pred_curr = y_pred_all[(i+0):(i+1)]
        if  y_pred_curr == 1 :# green bar prediction
            pointsCurr  = bar_range
            percentCurr = bar_range/open*100
            if  currBarUp == 1 :
                listWinners.append(pointsCurr)
                win_long    += 1#elif currBarUp<0
            else:
                listLosers.append(pointsCurr)
                lose_long   += 1
            listLongs.append(pointsCurr)

            print(' buy @', str(round(open,2)), ' exit @', str(round(close,2)), ' profit = ', pointsCurr)
        else:# red bar prediction
            pointsCurr  = -bar_range
            percentCurr = -bar_range/open*100
            if  currBarUp == 1  :
                listLosers.append(pointsCurr)
                lose_shrt    += 1
            else:
                listWinners.append(pointsCurr)
                win_short    += 1
            listShorts.append(pointsCurr)

            print(' sell @', str(round(open,2)), ' exit @', str(round(close,2)), ' profit = ', pointsCurr)


        #balanceTotal += profit
        listTrades.append(pointsCurr)
        listTradesPercent.append(percentCurr)
        #plot_live(np.cumsum(listTrades, dtype=float) , title="commulative profit over time", xlabel="trades",  ylabel="points")

        cumsum = np.cumsum(listTrades, dtype=float)

        #plot_live(cumsum, i)

        print('longs=', str(len(listLongs)), ' short=', str(len(listShorts)), ' gain_all=', str(len(listWinners)), ' loss_all=', str(len(listLosers)), ' profitCurr=', str(round(pointsCurr, 2)), ' profitTotal=', round(sum(listTrades), 2), ' profitShorts=', round(sum(listShorts, 2)), ' profitLongs=', round(sum(listLongs), 2))

    # If you want to see the full error list then print the following statement
    longs  = len(listLongs)
    shorts = len(listShorts)
    totalTrades  = len(listTrades)
    totalProfit  = sum(listTrades)
    profitLongs  = sum(listLongs)
    profitShorts = sum(listShorts)
    profitWinner = sum(listWinners)
    profitLosers = sum(listLosers)
    if longs==0:
        longs=1
    if shorts == 0:
        shorts=1
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
    print("  longs positions  : " + str(longs )    , '# , ', round(longs /totalTrades*100,2)          , '% of total, ' ,round(   profitLongs  ,2),'$, ',str(round(win_long/longs *100,2)) , '% won, largest=',str(round(max( listLongs),2))  , '$, smallest=',str(round(min(listLongs) ,2)))
    print("  shorts positions : " + str(shorts)    , '# , ', round(shorts/totalTrades*100,2)          , '% of total, ' ,round(   profitShorts ,2),'$, ',str(round(win_short/shorts*100,2)), '% won, largest=',str(round(max(listShorts),2))  , '$, smallest=',str(round(min(listShorts),2)))
    print("  winner positions : ", len(listWinners), '# , ', round(len(listWinners)/totalTrades*100,2), '% of total, ' ,round(sum(listWinners),2),'$')
    print("  loser  positions : " ,len( listLosers), '# , ', round(len(listLosers) /totalTrades*100,2), '% of total, ' ,round(sum(listLosers ),2),'$')
    plt.clf()
    plot_barchart(listTrades       , title="BT-trade points  over time", xlabel="trades",  ylabel="points")
    plt.clf()
    plot_barchart(listTradesPercent, title="BT-trade percent over time", xlabel="trades",  ylabel="percent", colors='blue')

    listTrades.insert(1,initialDeposit)
    plt.clf()
    plot_list(np.cumsum(listTrades, dtype=float)        , title="BT-commulative points over time", xlabel="trades",  ylabel="points")
    plt.clf()
    plot_list(np.cumsum(listTradesPercent, dtype=float) , title="BT-commulative percent over time", xlabel="trades",  ylabel="percent")


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
    plt.savefig('files/output/'+title+'.png')
    #print (listTrades)




print('\nBacktesting')
print('\n=========================================')
pd.set_option('display.max_columns', 500)
pd.set_option('display.width'      , 1000)

symbol      ='^GSPC'# ^GSPC = SP500 3600, DJI 300
skipRows    = 3600#3600 6600
epochs      = 600
size_hidden = 512
names_input   = ['nvo', 'mom5', 'mom10', 'mom20', 'mom50', 'sma10', 'sma20', 'sma50', 'sma200', 'sma400', 'range_sma', 'range_sma1', 'range_sma2', 'range_sma3', 'range_sma4', 'bb_hi10', 'bb_lo10', 'bb_hi20', 'bb_lo20', 'bb_hi50', 'bb_lo50', 'bb_hi200', 'bb_lo200', 'rel_bol_hi10', 'rel_bol_lo10', 'rel_bol_hi20', 'rel_bol_lo20', 'rel_bol_hi50', 'rel_bol_lo50', 'rel_bol_hi200', 'rel_bol_lo200', 'rsi10', 'rsi20', 'rsi50', 'rsi5', 'stoc10', 'stoc20', 'stoc50', 'stoc200']
names_output  = ['Green bar', 'Red Bar']#, 'Hold Bar']#Green bar', 'Red Bar', 'Hold Bar'
size_input   = len(names_input) # 39#x_train.shape[1] # no of features
size_output  = len(names_output)#  2 # there are 3 classes (buy.sell. hold) or (green,red,hold)
filename = 'mlpt_'+symbol+'_epc'+str(epochs)+'_hid'+str(size_hidden)+'_inp'+str(size_input)+'_out'+str(size_output)+'.model'


print('trying to backtest with model ',filename ,' and input values = ', names_input)
back_test(filename, symbol, skipRows, names_input, names_output, start_date='1970-01-03', end_date='2019-05-05')
