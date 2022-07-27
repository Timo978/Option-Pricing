import pandas as pd
import numpy as np
import datetime
import plotly as py
import plotly.graph_objects as go

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

from pandas_datareader import data
start = datetime.datetime(2017,1,1)#获取数据的时间段-起始时间
end = datetime.date.today()#获取数据的时间段-结束时间
df = data.get_data_yahoo('BTC-USD', start=start, end=end)
RV_cones(df,plot = True,crypto=True)