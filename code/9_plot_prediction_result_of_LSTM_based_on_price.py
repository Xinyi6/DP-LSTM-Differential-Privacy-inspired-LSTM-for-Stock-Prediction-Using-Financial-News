#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
# from wordcloud import WordCloud,STOPWORDS
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


# In[2]:


dataframe = pd.read_csv("D://NeurIPS Workshop/data/source_price.csv")
sp_ori=dataframe['Adj Close'][0:dataframe.shape[0]-1]
print(len(sp_ori))

sp_1dim_lstm=np.load('D://NeurIPS Workshop/stockdata/result2/sp_1dim_lstm.npy')
print(len(sp_1dim_lstm))

sp_5dim=np.load('D://NeurIPS Workshop/stockdata/result2/sp_5dim.npy')
print(len(sp_5dim))

sp_5dim_n01=np.load('D://NeurIPS Workshop/stockdata/result2/sp_5dim_n01.npy')
print(len(sp_5dim_n01))


# In[3]:


x_label = np.arange(111, 120)
x_label


# In[3]:


x_label = np.arange(93, 120)
x_label


# In[4]:


fig = plt.figure()
left, bottom, width, height = 0.1,0.1,1.8,0.7
ax1 = fig.add_axes([left,bottom,width,height])
plt.grid(linestyle = "--")

plt.plot(sp_ori,color="cornflowerblue",linestyle = "-",label="SP500",linewidth=1.5)
plt.plot(x_label,sp_1dim_lstm,color="hotpink",linestyle = "-",label="SP500 LSTM without news",linewidth=1.5)
plt.plot(x_label,sp_5dim,color="green",linestyle = "-.",label="SP500 LSTM with news",linewidth=1.5)
plt.plot(x_label,sp_5dim_n01,color="red",linestyle = "-",label="SP500 DP-LSTM",linewidth=1.5)
plt.yticks(fontsize=14)
plt.xlabel("Date",fontsize=14)
plt.ylabel("Price",fontsize=14)
plt.xlim(0,120)    
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=14) 

x=np.array([0.,  40.,  80., 120.])
group_labels=['12/07/2017','02/06/2018', '04/05/2018','06/01/2018']
plt.xticks(x,group_labels,fontsize=14,)
plt.savefig('D://NeurIPS Workshop/plot/sp3.jpg',dpi = 200,bbox_inches='tight')

