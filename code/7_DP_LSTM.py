#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from math import pi,sqrt,exp,pow,log
from numpy.linalg import det, inv
from abc import ABCMeta, abstractmethod
from sklearn import cluster

import statsmodels.api as sm 
import scipy.stats as scs
import scipy.optimize as sco
import scipy.interpolate as sci
from scipy import stats


# In[3]:


df = pd.read_csv("D://NeurIPS Workshop/data/source_price.csv")

wsj_var=np.var(df.wsj_mean_compound)
cnbc_var=np.var(df.cnbc_mean_compound)
fortune_var=np.var(df.fortune_mean_compound)
reuters_var=np.var(df.reuters_mean_compound)

mu=0

noise=0.1

sigma_wsj=noise*wsj_var
sigma_cnbc=noise*cnbc_var
sigma_fortune=noise*fortune_var
sigma_reuters=noise*reuters_var

n=df.shape[0]

df_noise=pd.DataFrame()

df_noise['wsj_noise']=df['wsj_mean_compound']
df_noise['cnbc_noise']=df['cnbc_mean_compound']
df_noise['fortune_noise']=df['fortune_mean_compound']
df_noise['reuters_noise']=df['reuters_mean_compound']

for i in range(0,n):
    df_noise['wsj_noise'][i]+=np.random.normal(mu,sigma_wsj)
    df_noise['cnbc_noise'][i]+=np.random.normal(mu,sigma_cnbc)
    df_noise['fortune_noise'][i]+=np.random.normal(mu,sigma_fortune)
    df_noise['reuters_noise'][i]+=np.random.normal(mu,sigma_reuters)
    

# df_noise.to_csv("D://NeurIPS Workshop/data/source_price_noise0.csv")
dfn=pd.read_csv("D://NeurIPS Workshop/data/source_price_noise0.csv",index_col=0)  

df_1n=pd.DataFrame()
df_1n['wsj']=dfn['wsj_noise']
df_1n['cnbc']=df['cnbc_mean_compound']
df_1n['fortune']=df['fortune_mean_compound']
df_1n['reuters']=df['reuters_mean_compound']
df_1n['price']=df['Adj Close']

df_2n=pd.DataFrame()
df_2n['wsj']=df['wsj_mean_compound']
df_2n['cnbc']=dfn['cnbc_noise']
df_2n['fortune']=df['fortune_mean_compound']
df_2n['reuters']=df['reuters_mean_compound']
df_2n['price']=df['Adj Close']

df_3n=pd.DataFrame()
df_3n['wsj']=df['wsj_mean_compound']
df_3n['cnbc']=df['cnbc_mean_compound']
df_3n['fortune']=dfn['fortune_noise']
df_3n['reuters']=df['reuters_mean_compound']
df_3n['price']=df['Adj Close']

df_4n=pd.DataFrame()
df_4n['wsj']=df['wsj_mean_compound']
df_4n['cnbc']=df['cnbc_mean_compound']
df_4n['fortune']=df['fortune_mean_compound']
df_4n['reuters']=dfn['reuters_noise']
df_4n['price']=df['Adj Close']

df1=df_1n
df2=df_2n
df3=df_3n
df4=df_4n


# In[4]:


split = (0.85)
sequence_length=10;
normalise= True
batch_size=100;
input_dim=5
input_timesteps=9
neurons=50
epochs=5
prediction_len=1
dense_output=1
drop_out=0


# In[5]:


i_split = int(len(df1) * split)

cols = ['price','wsj','cnbc','fortune','reuters']
data_train_1 = df1.get(cols).values[:i_split]
data_train_2 = df2.get(cols).values[:i_split]
data_train_3 = df3.get(cols).values[:i_split]
data_train_4 = df4.get(cols).values[:i_split]

len_train  = len(data_train_1)
len_train_windows = None


# In[6]:


#get_train_data 
data_windows = []
for i in range(len_train - sequence_length):
    data_windows.append(data_train_1[i:i+sequence_length])
data_windows = np.array(data_windows).astype(float)
  
window_data=data_windows
win_num=window_data.shape[0]
col_num=window_data.shape[2]

normalised_data = []
record_min=[]
record_max=[]

for win_i in range(0,win_num):
    normalised_window = []
    for col_i in range(0,1):#col_num):
      temp_col=window_data[win_i,:,col_i]
      temp_min=min(temp_col)
      if col_i==0:
        record_min.append(temp_min)#record min
      temp_col=temp_col-temp_min
      temp_max=max(temp_col)
      if col_i==0:
        record_max.append(temp_max)#record max
      temp_col=temp_col/temp_max
      normalised_window.append(temp_col)
    for col_i in range(1,col_num):
      temp_col=window_data[win_i,:,col_i]
      normalised_window.append(temp_col)
    normalised_window = np.array(normalised_window).T
    normalised_data.append(normalised_window)
normalised_data=np.array(normalised_data)

# normalised_data=window_data
data_windows=normalised_data
x_train1 = data_windows[:, :-1]
y_train1 = data_windows[:, -1,[0]]
print('x_train1.shape',x_train1.shape)
print('y_train1.shape',y_train1.shape)


# In[7]:


#get_train_data 
data_windows = []
for i in range(len_train - sequence_length):
    data_windows.append(data_train_2[i:i+sequence_length])
data_windows = np.array(data_windows).astype(float)
  
window_data=data_windows
win_num=window_data.shape[0]
col_num=window_data.shape[2]

normalised_data = []
record_min=[]
record_max=[]

for win_i in range(0,win_num):
    normalised_window = []
    for col_i in range(0,1):#col_num):
      temp_col=window_data[win_i,:,col_i]
      temp_min=min(temp_col)
      if col_i==0:
        record_min.append(temp_min)#record min
      temp_col=temp_col-temp_min
      temp_max=max(temp_col)
      if col_i==0:
        record_max.append(temp_max)#record max
      temp_col=temp_col/temp_max
      normalised_window.append(temp_col)
    for col_i in range(1,col_num):
      temp_col=window_data[win_i,:,col_i]
      normalised_window.append(temp_col)
    normalised_window = np.array(normalised_window).T
    normalised_data.append(normalised_window)
normalised_data=np.array(normalised_data)

# normalised_data=window_data
data_windows=normalised_data
x_train2 = data_windows[:, :-1]
y_train2 = data_windows[:, -1,[0]]
print('x_train2.shape',x_train2.shape)
print('y_train2.shape',y_train2.shape)


# In[8]:


#get_train_data 
data_windows = []
for i in range(len_train - sequence_length):
    data_windows.append(data_train_3[i:i+sequence_length])
data_windows = np.array(data_windows).astype(float)
  
window_data=data_windows
win_num=window_data.shape[0]
col_num=window_data.shape[2]

normalised_data = []
record_min=[]
record_max=[]

for win_i in range(0,win_num):
    normalised_window = []
    for col_i in range(0,1):#col_num):
      temp_col=window_data[win_i,:,col_i]
      temp_min=min(temp_col)
      if col_i==0:
        record_min.append(temp_min)#record min
      temp_col=temp_col-temp_min
      temp_max=max(temp_col)
      if col_i==0:
        record_max.append(temp_max)#record max
      temp_col=temp_col/temp_max
      normalised_window.append(temp_col)
    for col_i in range(1,col_num):
      temp_col=window_data[win_i,:,col_i]
      normalised_window.append(temp_col)
    normalised_window = np.array(normalised_window).T
    normalised_data.append(normalised_window)
normalised_data=np.array(normalised_data)

# normalised_data=window_data
data_windows=normalised_data
x_train3 = data_windows[:, :-1]
y_train3 = data_windows[:, -1,[0]]
print('x_train3.shape',x_train3.shape)
print('y_train3.shape',y_train3.shape)


# In[9]:


#get_train_data 

data_windows = []
for i in range(len_train - sequence_length):
    data_windows.append(data_train_4[i:i+sequence_length])
data_windows = np.array(data_windows).astype(float)
  
window_data=data_windows
win_num=window_data.shape[0]
col_num=window_data.shape[2]

normalised_data = []
record_min=[]
record_max=[]

for win_i in range(0,win_num):
    normalised_window = []
    for col_i in range(0,1):#col_num):
      temp_col=window_data[win_i,:,col_i]
      temp_min=min(temp_col)
      if col_i==0:
        record_min.append(temp_min)#record min
      temp_col=temp_col-temp_min
      temp_max=max(temp_col)
      if col_i==0:
        record_max.append(temp_max)#record max
      temp_col=temp_col/temp_max
      normalised_window.append(temp_col)
    for col_i in range(1,col_num):
      temp_col=window_data[win_i,:,col_i]
      normalised_window.append(temp_col)
    normalised_window = np.array(normalised_window).T
    normalised_data.append(normalised_window)
normalised_data=np.array(normalised_data)

# normalised_data=window_data
data_windows=normalised_data
x_train4 = data_windows[:, :-1]
y_train4 = data_windows[:, -1,[0]]
print('x_train4.shape',x_train4.shape)
print('y_train4.shape',y_train4.shape)


# In[32]:


##concat train window
type(x_train4)


# In[10]:


x_train_t=np.concatenate((x_train1,x_train2,x_train3,x_train4),axis=0)
print(x_train_t.shape)
x_train=x_train_t

y_train_t=np.concatenate((y_train1,y_train2,y_train3,y_train4),axis=0)
print(y_train_t.shape)
y_train=y_train_t


# In[11]:


dataframe= pd.read_csv("D://NeurIPS Workshop/data/source_price.csv")
dataframe.columns=['date','wsj','cnbc','fortune','reuters','price']
cols = ['price','wsj','cnbc','fortune','reuters']
len_dataframe=dataframe.shape[0]

i_split = int(len(dataframe) * split)
data_test  = dataframe.get(cols).values[i_split:]

len_test   = len(data_test)
len_train_windows = None

print('data_test.shape',data_test.shape)


# In[12]:


data_windows = []
for i in range(len_test - sequence_length):
    data_windows.append(data_test[i:i+sequence_length])

data_windows = np.array(data_windows).astype(float)

y_test_ori = data_windows[:, -1, [0]]
print('y_test_ori.shape',y_test_ori.shape)


# In[13]:


y_test_ori


# In[14]:


window_data=data_windows
win_num=window_data.shape[0]
col_num=window_data.shape[2]
normalised_data = []
record_min=[]
record_max=[]

#normalize
for win_i in range(0,win_num):
    normalised_window = []
    for col_i in range(0,1):#col_num):
      temp_col=window_data[win_i,:,col_i]
      temp_min=min(temp_col)
      if col_i==0:
        record_min.append(temp_min)#record min
      temp_col=temp_col-temp_min
      temp_max=max(temp_col)
      if col_i==0:
        record_max.append(temp_max)#record max
      temp_col=temp_col/temp_max
      normalised_window.append(temp_col)
    for col_i in range(1,col_num):
      temp_col=window_data[win_i,:,col_i]
      normalised_window.append(temp_col)
    normalised_window = np.array(normalised_window).T
    normalised_data.append(normalised_window)
normalised_data=np.array(normalised_data)

# normalised_data=window_data
data_windows=normalised_data#get_test_data
x_test = data_windows[:, :-1]
y_test = data_windows[:, -1, [0]]
print('x_test.shape',x_test.shape)
print('y_test.shape',y_test.shape)


# In[15]:


# LSTM MODEL
model = Sequential()
model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences = True))
model.add(Dropout(drop_out))
model.add(LSTM(neurons,return_sequences = True))
model.add(LSTM(neurons,return_sequences =False))
model.add(Dropout(drop_out))
model.add(Dense(dense_output, activation='linear'))
# Compile model
model.compile(loss='mean_squared_error',
                optimizer='adam')
# Fit the model
model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size)


# In[16]:


#multi sequence predict
data=x_test
prediction_seqs = []
window_size=sequence_length
pre_win_num=int(len(data)/prediction_len)

for i in range(0,pre_win_num):
    curr_frame = data[i*prediction_len]
    predicted = []
    for j in range(0,prediction_len):
      temp=model.predict(curr_frame[newaxis,:,:])[0]
      predicted.append(temp)
      curr_frame = curr_frame[1:]
      curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
    prediction_seqs.append(predicted)


# In[17]:


#de_predicted
de_predicted=[]
len_pre_win=int(len(data)/prediction_len)
len_pre=prediction_len

m=0
for i in range(0,len_pre_win):
    for j in range(0,len_pre):
      de_predicted.append(prediction_seqs[i][j][0]*record_max[m]+record_min[m])
      m=m+1
print(de_predicted)

# np.save('D://NeurIPS Workshop/stockdata/result2/sp_5dim_n01_7030.npy',de_predicted)


# In[18]:


error = []
diff=y_test.shape[0]-prediction_len*pre_win_num

for i in range(y_test_ori.shape[0]-diff):
    error.append(y_test_ori[i,] - de_predicted[i])
    
squaredError = []
absError = []
for val in error:
    squaredError.append(val * val) 
    absError.append(abs(val))

error_percent=[]
for i in range(len(error)):
    val=absError[i]/y_test_ori[i,]
    val=abs(val)
    error_percent.append(val)

mean_error_percent=sum(error_percent) / len(error_percent)
accuracy=1-mean_error_percent

MSE=sum(squaredError) / len(squaredError)

print("MSE",MSE)
print('accuracy',accuracy)
print('mean_error_percent',mean_error_percent)

