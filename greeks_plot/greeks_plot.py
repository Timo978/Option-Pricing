import plotly as py
import plotly.graph_objects as go
from py_vollib_vectorized import get_all_greeks
import numpy as np

#Theta
#K
theta_k = []

for i in np.arange(30000,50000,100):
    theta_k.append((get_all_greeks('c',40000,int(i),0.25,0,0.3)['theta'])[0])

trace = go.Scatter(x=np.arange(30000,50000,100),
                   y=theta_k,
                   mode='lines',
                   name='theta_byK')
data = [trace]
py.offline.plot(data,filename='./theta_byK.html')

#S
theta_s = []

for i in np.arange(30000,50000,100):
    theta_s.append((get_all_greeks('c',int(i),40000,0.25,0,0.3)['theta'])[0])

trace = go.Scatter(x=np.arange(30000,50000,100),
                   y=theta_s,
                   mode='lines',
                   name='theta_byS')
data = [trace]
py.offline.plot(data,filename='./theta_byS.html')

#ttm
theta_itm = []
theta_atm = []
theta_otm = []

for i in np.arange(0.01,0.3,0.01):
    theta_itm.append((get_all_greeks('c', 40000, 30000, i, 0, 0.3)['theta'])[0])
    theta_atm.append((get_all_greeks('c', 40000, 40000, i, 0, 0.3)['theta'])[0])
    theta_otm.append((get_all_greeks('c',40000,70000,i,0,0.3)['theta'])[0])

trace = go.Scatter(x=np.arange(0.01,0.3,0.01),
                   y=theta_itm,
                   mode='lines',
                   name='theta_byTTM_itm')
trace1 = go.Scatter(x=np.arange(0.01,0.3,0.01),
                   y=theta_atm,
                   mode='lines',
                   name='theta_byTTM_atm')
trace2 = go.Scatter(x=np.arange(0.01,0.3,0.01),
                   y=theta_otm,
                   mode='lines',
                   name='theta_byTTM_otm')

data = [trace,trace1,trace2]
py.offline.plot(data,filename='./theta_byTTM2.html')


