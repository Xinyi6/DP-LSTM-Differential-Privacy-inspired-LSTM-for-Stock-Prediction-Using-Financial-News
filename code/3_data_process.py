#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv("D://NeurIPS Workshop/data/us_financial_news_articles_2018_with_sentiment.csv",index_col=0)


# In[4]:


df.head()


# In[10]:


df2=df.copy()
df2=pd.DataFrame(df2)
n=df.shape[0]

for i in range(0,n):
  df2['published_date'][i]=df2['published_date'][i][0:10]
  if i%10000==0:

    print(i)
df2.to_csv('D://NeurIPS Workshop/data/sentiment.csv')


# In[15]:


df2['source_name'].unique()


# In[24]:


grouped=df2.groupby(['source_name']).size()
print(grouped)


# In[29]:


# grouped=df2.groupby(['source_name']).filter(lambda x: x=='wsj.com')
# print(grouped)
# type(df2['source_name'][0])
dff1=df2[(df2.source_name=='wsj.com')]
print(dff1.shape)
dff2=df2[(df2.source_name=='cnbc.com')]
print(dff2.shape)
dff3=df2[(df2.source_name=='fortune.com')]
print(dff3.shape)
dff4=df2[(df2.source_name=='reuters.com')]
print(dff4.shape)

dff1.to_csv('D://NeurIPS Workshop/data/sentiment_wsj.csv')
dff2.to_csv('D://NeurIPS Workshop/data/sentiment_cnbc.csv')
dff3.to_csv('D://NeurIPS Workshop/data/sentiment_fortune.csv')
dff4.to_csv('D://NeurIPS Workshop/data/sentiment_reuters.csv')


# In[32]:


dff1.head()


# In[36]:


sp = pd.read_csv("D://NeurIPS Workshop/data/SP500.csv")#,index_col=0)
sp=pd.DataFrame(sp)
sp.shape


# In[37]:


#group by day, mean 
dff1g=dff1.groupby(['published_date']).agg(['mean'])
print(dff1g.shape)
dff2g=dff2.groupby(['published_date']).agg(['mean'])
print(dff2g.shape)
dff3g=dff3.groupby(['published_date']).agg(['mean'])
print(dff3g.shape)
dff4g=dff4.groupby(['published_date']).agg(['mean'])
print(dff4g.shape)

dff1g.to_csv('D://NeurIPS Workshop/data/sentiment_wsj_groupday.csv')
dff2g.to_csv('D://NeurIPS Workshop/data/sentiment_cnbc_groupday.csv')
dff3g.to_csv('D://NeurIPS Workshop/data/sentiment_fortune_groupday.csv')
dff4g.to_csv('D://NeurIPS Workshop/data/sentiment_reuters_groupday.csv')


# In[40]:


dff1g.columns


# In[37]:


d1 = pd.read_csv("D://NeurIPS Workshop/data/sentiment_wsj_groupday1.csv")#,index_col=0)
d2 = pd.read_csv("D://NeurIPS Workshop/data/sentiment_cnbc_groupday1.csv")
d3 = pd.read_csv("D://NeurIPS Workshop/data/sentiment_fortune_groupday1.csv")
d4 = pd.read_csv("D://NeurIPS Workshop/data/sentiment_reuters_groupday1.csv")
d1=pd.DataFrame(d1)
d1.head()


# In[5]:


print(type(sp['Date'][0]))
print(type(d1['published_date'][0]))
print(sp.head())
print(d1.head())


# In[59]:


sp1=sp.copy()
m=sp.shape[0]

for i in range (0,3):
    print(i)
    sp1['Date'][i].replace('-','/')


# In[64]:


sp1['Date'][0]==d1['published_date'][0]
print(sp1['Date'][0])
print(d1['published_date'][0])


# In[38]:


import time
t=d1['published_date'][0]
timeStruct = time.strptime(t, "%Y/%m/%d") 
strTime = time.strftime("%Y-%m-%d", timeStruct) 
print(strTime)

for i in range(0,d1.shape[0]):
    t=d1['published_date'][i]
    timeStruct = time.strptime(t, "%Y/%m/%d") 
    d1['published_date'][i] = time.strftime("%Y-%m-%d", timeStruct) 
    
d1.head()
    


# In[39]:


for i in range(0,d2.shape[0]):
    t=d2['published_date'][i]
    timeStruct = time.strptime(t, "%Y/%m/%d") 
    d2['published_date'][i] = time.strftime("%Y-%m-%d", timeStruct) 
    
for i in range(0,d3.shape[0]):
    t=d3['published_date'][i]
    timeStruct = time.strptime(t, "%Y/%m/%d") 
    d3['published_date'][i] = time.strftime("%Y-%m-%d", timeStruct) 
    
for i in range(0,d4.shape[0]):
    t=d4['published_date'][i]
    timeStruct = time.strptime(t, "%Y/%m/%d") 
    d4['published_date'][i] = time.strftime("%Y-%m-%d", timeStruct) 


# In[17]:


# d1.to_csv("D://NeurIPS Workshop/data/sentiment_wsj_groupday2.csv")
# d2.to_csv('D://NeurIPS Workshop/data/sentiment_cnbc_groupday2.csv')
# d3.to_csv('D://NeurIPS Workshop/data/sentiment_fortune_groupday2.csv')
# d4.to_csv('D://NeurIPS Workshop/data/sentiment_reuters_groupday2.csv')

np.save('D://NeurIPS Workshop/data/d1.npy',d1)
np.save('D://NeurIPS Workshop/data/d2.npy',d2)
np.save('D://NeurIPS Workshop/data/d3.npy',d3)
np.save('D://NeurIPS Workshop/data/d4.npy',d4)


# In[30]:


d1.head()


# In[74]:


date_union_3=pd.DataFrame(columns=('idx','date','mean_compound','comp_flag'))
sp_len=sp.shape[0]
d_len=d3.shape[0]
d=d3.copy()
for i in range(0,sp_len):
    idx=i
    date=sp['Date'][i]
    j=0
    t=0
    while j<d_len:
        if sp['Date'][i]==d['published_date'][j]:
            mean_compound=d['compound'][j]
            comp_flag=1
            t=1
            break
        j=j+1
    if t==0:
        mean_compound=0
        comp_flag=0
    
    date_union_3=date_union_3.append(pd.DataFrame({'idx':[idx],
                                                  'date':[date],
                                                  'mean_compound':[mean_compound],
                                                  'comp_flag':[comp_flag]
        
    }),ignore_index=True)


# In[77]:


date_union_1=pd.DataFrame(columns=('idx','date','mean_compound','comp_flag'))
sp_len=sp.shape[0]
d_len=d1.shape[0]
d=d1.copy()
for i in range(0,sp_len):
    idx=i
    date=sp['Date'][i]
    j=0
    t=0
    while j<d_len:
        if sp['Date'][i]==d['published_date'][j]:
            mean_compound=d['compound'][j]
            comp_flag=1
            t=1
            break
        j=j+1
    if t==0:
        mean_compound=0
        comp_flag=0
    
    date_union_1=date_union_1.append(pd.DataFrame({'idx':[idx],
                                                  'date':[date],
                                                  'mean_compound':[mean_compound],
                                                  'comp_flag':[comp_flag]
        
    }),ignore_index=True)


# In[83]:


date_union_4=pd.DataFrame(columns=('idx','date','mean_compound','comp_flag'))
sp_len=sp.shape[0]
d_len=d4.shape[0]
d=d4.copy()
for i in range(0,sp_len):
    idx=i
    date=sp['Date'][i]
    j=0
    t=0
    while j<d_len:
        if sp['Date'][i]==d['published_date'][j]:
            mean_compound=d['compound'][j]
            comp_flag=1
            t=1
            break
        j=j+1
    if t==0:
        mean_compound=0
        comp_flag=0
    
    date_union_4=date_union_4.append(pd.DataFrame({'idx':[idx],
                                                  'date':[date],
                                                  'mean_compound':[mean_compound],
                                                  'comp_flag':[comp_flag]
        
    }),ignore_index=True)


# In[86]:


date_union_4.shape


# In[68]:


date_union_2=pd.DataFrame(columns=('idx','date','mean_compound','comp_flag'))
sp_len=sp.shape[0]
d_len=d2.shape[0]
d=d2.copy()
for i in range(0,sp_len):
    idx=i
    date=sp['Date'][i]
    j=0
    t=0
    while j<d_len:
        if sp['Date'][i]==d['published_date'][j]:
            mean_compound=d2['compound'][j]
            comp_flag=1
            t=1
            break
        j=j+1
    if t==0:
        mean_compound=0
        comp_flag=0
    
    date_union_2=date_union_2.append(pd.DataFrame({'idx':[idx],
                                                  'date':[date],
                                                  'mean_compound':[mean_compound],
                                                  'comp_flag':[comp_flag]
        
    }),ignore_index=True)


# In[69]:


d2.head()


# In[72]:


date_union_2.shape


# In[57]:


date_union_3=pd.DataFrame(columns=('idx','date','mean_compound','comp_flag'))
sp_len=sp.shape[0]
for i in range(0,sp_len):
    idx=i
    date=sp['Date'][i]
    for j in range(0,d3.shape[0]):
        if sp['Date'][i]==d3['published_date'][j]:
            mean_compound=d3['compound'][j]
            comp_flag=1
        elif j<d3.shape[0]:
            continue
        else:
            mean_compound=0
            comp_flag=0
    
    date_union_3=date_union_3.append(pd.DataFrame({'idx':[idx],
                                                  'date':[date],
                                                  'mean_compound':[mean_compound],
                                                  'comp_flag':[comp_flag]
        
    }),ignore_index=True)


# In[59]:


date_union_2


# In[58]:


d2


# In[46]:


print(d1['published_date'][0])
print(sp['Date'][0])
print(d1['published_date'][0]==sp['Date'][0])


# In[52]:


date_union_1.shape


# In[87]:


date_union_3.to_csv("D://NeurIPS Workshop/data/date_union_3.csv")
date_union_2.to_csv('D://NeurIPS Workshop/data/date_union_2.csv')
date_union_1.to_csv('D://NeurIPS Workshop/data/date_union_1.csv')
date_union_4.to_csv('D://NeurIPS Workshop/data/sdate_union_4.csv')

np.save('D://NeurIPS Workshop/data/date_union_1.npy',date_union_1)
np.save('D://NeurIPS Workshop/data/date_union_2.npy',date_union_2)
np.save('D://NeurIPS Workshop/data/date_union_3.npy',date_union_3)
np.save('D://NeurIPS Workshop/data/date_union_4.npy',date_union_4)


# In[88]:


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


# In[147]:


data = pd.read_csv("D://NeurIPS Workshop/data/source_price.csv",index_col=0)
data.head(50)

