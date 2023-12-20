#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[6]:


df = pd.read_csv("C:\\Users\\gajendra singh\\OneDrive\\Desktop\\pandas\\market_basket.csv", header = None)


# In[7]:


df.head()


# In[8]:


df.shape


# In[10]:


# data Processing
transaction = []
for i in range(0,7501):
    transaction.append([str(df.values[i,j]) for j in range(0,20)])
    
    
# training the data
from apyori import apriori
rules = apriori(transaction,
               min_support = 0.003,
               min_confidence = 0.2,
               min_left = 3,
               min_length = 2)


# In[11]:


a = list(rules)
result = [list(a[i][0]) for i in range(0, len(a))]


# In[12]:


a


# In[13]:


result


# In[ ]:




