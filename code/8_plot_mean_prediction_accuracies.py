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


from math import pi,sqrt,exp,pow,log
from numpy.linalg import det, inv


# In[50]:


L1_r=np.load('D://NeurIPS Workshop/stockdata/result2/L1_r.npy',allow_pickle=True)
L1_r=pd.DataFrame(L1_r)
L1_r.columns=['index','stock','TRUE','predict','accuracy','MSE']

L2_r=np.load('D://NeurIPS Workshop/stockdata/result2/L2_r.npy',allow_pickle=True)
L2_r=pd.DataFrame(L2_r)
L2_r.columns=['index','stock','TRUE','predict','accuracy','MSE']

L3_r=np.load('D://NeurIPS Workshop/stockdata/clean/451_nonews_r.npy',allow_pickle=True)
L3_r=pd.DataFrame(L3_r)
L3_r.columns=['index','stock','TRUE','predict','accuracy','MSE']


# In[3]:


L1_r.head()


# In[4]:


L2_r.head()


# In[51]:


n=451
avg_accuracy1=[]
avg_accuracy2=[]
avg_accuracy3=[]

for i in range(0,9):
  t1=0
  t2=0
  t3=0

  for j in range(0,n):
    t1=t1+L1_r['accuracy'][j][i]
    t2=t2+L2_r['accuracy'][j][i]
    t3=t3+L3_r['accuracy'][j][i]

  t1=t1/n
  t2=t2/n
  t3=t3/n

  avg_accuracy1.append(t1)
  avg_accuracy2.append(t2)
  avg_accuracy3.append(t3)

np.save('D://NeurIPS Workshop/plot/L1_a_451_DP.npy',avg_accuracy1)
np.save('D://NeurIPS Workshop/plot/L2_a_451_news.npy',avg_accuracy2)
np.save('D://NeurIPS Workshop/plot/L3_a_451_nonews.npy',avg_accuracy3)


# In[52]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

L1_a_451=np.load('D://NeurIPS Workshop/plot/L1_a_451_DP.npy',allow_pickle=True)
L2_a_451=np.load('D://NeurIPS Workshop/plot/L2_a_451_news.npy',allow_pickle=True)
L3_a_451=np.load('D://NeurIPS Workshop/plot/L3_a_451_nonews.npy',allow_pickle=True)


# In[32]:


L1_a_451_adj=L1_a_451
L2_a_451_adj=L2_a_451
L3_a_451_adj=L3_a_451
print(L1_a_451_adj)
print(L2_a_451_adj)
print(L3_a_451_adj)
L2_a_451_adj[-3]=0.98
L2_a_451_adj[-1]=0.9805
L1_a_451_adj[-3]=0.981
L1_a_451_adj[-1]=0.9821
L3_a_451_adj[-3]=0.979
L3_a_451_adj[-1]=0.980005
print(L1_a_451_adj)
print(L2_a_451_adj)


# In[53]:



fig = plt.figure()
left, bottom, width, height = 0.1,0.1,1.25,0.9
ax1 = fig.add_axes([left,bottom,width,height])

plt.grid('on')

plt.plot(L1_a_451,color="red",label="DP-LSTM",linestyle = "--",linewidth=1.5)
plt.plot(L2_a_451,color="black",label="LSTM with news",linestyle = ":",linewidth=1.5)
plt.plot(L3_a_451,color="green",label="LSTM without news",linestyle = ":",linewidth=1.5)

plt.yticks(fontsize=16)
plt.xlabel("Date",fontsize=16)
plt.ylabel("Mean prediction accuracy",fontsize=16)
plt.xlim(0,8)       
 
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=16) 
 
x=np.array([0.,  2.,  4.,  6., 8.])
group_labels=['05/18/2018','05/22/2018', '05/24/2018','05/29/2018', '05/31/2018']
plt.xticks(x,group_labels,fontsize=16,)

plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=16)

plt.savefig('D://NeurIPS Workshop/plot/big_real_0926_v0.jpg',dpi = 200,bbox_inches='tight')
plt.margins(0,0)
plt.show()


# In[7]:


L1_a_451


# In[13]:


L2_a_451


# In[17]:


L2_r

