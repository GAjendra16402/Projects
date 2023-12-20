#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


import requests
from bs4 import BeautifulSoup


# In[5]:


headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
response = requests.get('https://www.imdb.com/chart/top/?ref_=nv_mv_250',headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')
title = soup.find_all("h3",{"class":"ipc-title__text"})
rating = soup.find_all("span",{"class":"ipc-rating-star--base ipc-rating-star--imdb ratingGroup--imdb-rating"})


# In[17]:


# title_list = []

# for i in range(1,251):
#     title_list.append(title[i].text)
    
# for i in range(0,250):
#     print(title_list[i]," ", rating[i].text
rating


# In[25]:


df = pd.read_csv("C:\\Users\\gajendra singh\\Downloads\\movie_list.csv")


# In[30]:


df['rating']


# In[33]:


df['rating'] = df['rating'].str.lower()


# In[34]:


df['rating'][3]


# In[35]:


import re
def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'',text)


# In[36]:


df['rating'] = df['rating'].apply(remove_html_tags)


# In[37]:


df['rating'][3]


# In[38]:


import string
string.punctuation


# In[39]:


exclude = string.punctuation


# In[40]:


def remove_punc(text):
    for char in exclude:
        text = text.replace(char, '')
    return text


# In[41]:


text = '9.0\xa0(1.3m)'
print(remove_punc(text))


# In[42]:


def remove_stopwords(text):
    new_text = []
    
    for word in text.split():
        if word in set(stopwords.words('english')):
            new_text.append('')
            
        else:
            new_text.append(word)

    x = new_text[:]
    new_text.clear()
    return "".join(x)


# In[45]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[46]:


STOPWORDS = set(stopwords.words('english'))


# In[47]:


df['rating'] = df['rating'].apply(remove_stopwords)


# In[49]:


df['rating']


# In[50]:


df


# In[51]:


df['title'] = df['title'].str.replace(r'^\d+\.\s+', '', regex=True)


# In[52]:


df


# In[53]:


df['votings'] = df['rating'].str.extract(r'\(([\d,]+)k\)', expand=True)


# In[59]:


df


# In[60]:


df.info()


# In[ ]:




