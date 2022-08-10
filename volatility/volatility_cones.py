import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas_datareader import data

start = datetime.datetime(2017,1,1)#获取数据的时间段-起始时间
end = datetime.date.today()#获取数据的时间段-结束时间
df = data.get_data_yahoo('BTC-USD', start=start, end=end)

def RV_cones(df,plot,crypto):
    max = []
    m75 = []
    m50 = []
    m25 = []

    df['log_ret'] = np.log(df['Close']/df['Close'].shift())
    x = np.arange(20, 300, 20)
    for i in x:
        df[f'rv_{i}'] = df['log_ret'].rolling(i).std() * np.sqrt(365) if crypto == True else df['log_ret'].rolling(i).std() * np.sqrt(252)
        max.append(df[f'rv_{i}'].quantile(1))
        m75.append(df[f'rv_{i}'].quantile(.75))
        m50.append(df[f'rv_{i}'].quantile(.5))
        m25.append(df[f'rv_{i}'].quantile(.25))
    if plot == True:
        trace0 = go.Scatter(x=x,
                            y=m25,
                            mode='markers+lines',
                            name='25%')
        trace1 = go.Scatter(x=x,
                            y=m50,
                            mode='markers+lines',
                            name='50%')
        trace2 = go.Scatter(x=x,
                            y=m75,
                            mode='markers+lines',
                            name='75%')
        trace3 = go.Scatter(x=x,
                            y=max,
                            mode='markers+lines',
                            name='max')
        data = [trace0, trace1, trace2, trace3]
        py.offline.plot(data, filename='./rv_cone.html')
    else:
        return np.array([m25,m50,m75,max])

RV_cones(df,plot = True,crypto=True)

def rv_quantile(df,plot,crypto):
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift())
    x = [5,10,20]
    for i in x:
        df[f'rv_{i}'] = df['log_ret'].rolling(i).std() * np.sqrt(365) if crypto == True else df['log_ret'].rolling(i).std() * np.sqrt(252)
        df[f'rv_{i}_25'] = df[f'rv_{i}'].rolling(i).quantile(.25)
        df[f'rv_{i}_50'] = df[f'rv_{i}'].rolling(i).quantile(.5)
        df[f'rv_{i}_75'] = df[f'rv_{i}'].rolling(i).quantile(.75)
        df[f'rv_{i}_100'] = df[f'rv_{i}'].rolling(i).quantile(1)
        trace = go.Scatter()

    if plot == True:
        plt.figure(figsize=(28, 20))

        plt.subplot(3, 1, 1)
        plt.plot(df[['rv_5_25', 'rv_5_50', 'rv_5_75', 'rv_5_100']])
        plt.title("RV_5 percentile")

        plt.subplot(3, 1, 2)
        plt.plot(df[['rv_10_25', 'rv_10_50', 'rv_10_75', 'rv_10_100']])
        plt.title("RV_10 percentile")

        plt.subplot(3, 1, 3)
        plt.plot(df[['rv_20_25', 'rv_20_50', 'rv_20_75', 'rv_20_100']])
        plt.title("RV_20 percentile")

    # plt.savefig(f'./rv_quantile.png')
    plt.show()

rv_quantile(df,plot = True,crypto = True)