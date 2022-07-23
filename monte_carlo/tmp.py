import datetime
import pandas as pd
import numpy as np

def ts2date(ts):
    dt = datetime.datetime.fromtimestamp(ts)
    dt = datetime.datetime.strftime(dt,"%Y-%m-%dT%H:%M:%S")
    return dt

df = pd.read_csv("~/Desktop/BTC220603-1min-2022-06-01.csv",header=False,index_col=[['datetime','open','high','low','close']])
df.columns = [['datetime','open','high','low','close']]

df.datetime = df.apply(lambda x: ts2date(x['datetime']),axis=1)
df['log_ret'] = np.log(df["close"]/df["close"].shift(1))
CCHV = np.sum(df['log_ret']**2)/len(df)
volatility1 = df.log_ret.rolling(window=1).std(ddof=0)*np.sqrt(365)
RV=((np.log(df['close'])-np.log(df['close'].shift())).dropna()**2).sum()

