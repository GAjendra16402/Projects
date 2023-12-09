#!/usr/bin/env python
# coding: utf-8

# # Real Data Project

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("C:\\Users\\gajendra singh\\OneDrive\\Desktop\\pandas\\Real data project\\hou_all.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


df.describe()


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


df.hist(bins = 50, figsize=(20,15))
plt.show()


# In[9]:


train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)


# In[10]:


train_set


# In[11]:


test_set


# In[12]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['CHAS']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]


# In[13]:


strat_test_set['CHAS'].value_counts()


# In[14]:


strat_train_set['CHAS'].value_counts()


# In[15]:


df = strat_train_set.copy()


# # Looking for correlations

# In[16]:


corr_matrix = df.corr()


# In[17]:


corr_matrix['MEDV'].sort_values(ascending=False) # strong positive correlation


# In[18]:


from pandas.plotting import scatter_matrix
attributes = ['MEDV','RM',"ZN",'LSTAT']
scatter_matrix(df[attributes], figsize=(12,8))


# In[19]:


df.plot(kind='scatter', x='RM', y='MEDV', alpha=0.8)


# # attributes combination 

# In[20]:


df['TAXRM'] = df['TAX']/df['RM']


# In[21]:


df.head()


# In[22]:


corr_matrix = df.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[23]:


df.plot(kind='scatter', x='TAXRM',y='MEDV',alpha=0.8)


# In[24]:


df = strat_train_set.drop('MEDV', axis =1)
df_labels = strat_train_set['MEDV'].copy()


# # missing attributes

# In[25]:


# to take care of missing attributes , you have three options:
#     1. get rid of the missing values.
#     2. get rid of the whole attribution
#     3. set the value to some value(0, mean or median)


# In[26]:


# option no.3
   
median = df['RM'].median()


# In[27]:


df['RM'].fillna(median)


# In[28]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(df)


# In[29]:


imputer.statistics_


# In[30]:


X = imputer.transform(df)


# In[31]:


df_new = pd.DataFrame(X, columns = df.columns)


# In[32]:


df_new.describe()


# # Scikit-learn Design

# # primarly, three types of objects
# 1. Estimators - It estimate some parameter based on a dataset. ex. imputer
#     it has a fit method and transform method.
#     
#     fit method - Fits the dataset and calculates internal parameters
# 2. Transforms - transform method takes input and returns output based on the learning from fit(). 
#                 It also has a convenience function called fit_transform() which fits and then transforms.
#     
# 3. Preictors - LinearRegression model is an example of predictor. fit() and predict
#                 are two common functions. It also gives score() fuction which will evaluate the predictions.
# 

# # Feature Scaling

# Primarly, two types of feature scaling methods:
# 
# 1. Min-max scaling(normalization)
#      (value - min)/(max - min)
#      sklearn provides a class called MinMaxScaler for this
#  
# 2. Standardization
#     (value - mean)/std
#     Sklearn provides a class called standardScaler for this 

# # CREATING PIPELINE

# In[33]:


from sklearn.pipeline import Pipeline


# In[34]:


from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])


# In[35]:


df_num_new = my_pipeline.fit_transform(df_new)


# In[36]:


df_num_new.shape


# # Selecting a desired model for Dragon real estates

# In[66]:


#from sklearn.linear_model import LinearRegression
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()


# In[67]:


model.fit(df_num_new, df_labels)


# In[68]:


some_data = df.iloc[:5]


# In[69]:


some_labels = df_labels.iloc[:5]


# In[70]:


prepared_data = my_pipeline.transform(some_data)


# In[71]:


model.predict(prepared_data)


# In[72]:


list(some_labels)


# # Evaluating the model

# In[73]:


from sklearn.metrics import mean_squared_error
df_predictions = model.predict(df_num_new)
mse = mean_squared_error(df_labels, df_predictions)
rmse = np.sqrt(mse)


# # Now data is overfitted so we have to solve that

# In[74]:


rmse


# # Using Better evaluation technique - Cross Validaion

# In[75]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, df_num_new, df_labels, scoring = 'neg_mean_squared_error', cv = 10)
rmse_scores = np.sqrt(-scores)


# In[76]:


rmse_scores


# In[77]:


def print_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("standard Deviation:", scores.std())


# In[78]:


print_scores(rmse_scores)


# # Saving the model 

# In[79]:


from joblib import dump, load
dump(model, 'Dragon.joblib')


# # Testing the model on the test data

# In[88]:


X_test = strat_test_set.drop('MEDV', axis = 1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(Y_test))


# In[85]:


final_rmse


# In[89]:


prepared_data[0]


# # Using the model

# In[91]:


from joblib import dump, load
import numpy as np
model = load('Dragon.joblib')
input = np.array([[-0.43942006,  0.12628155, -1.12165014, -0.27288841, 0.42262747,
       1.24323182, -1.31238772,  0.61111401, -1.0016859 , -0.5778192 ,
       1.97491834,  1.41164221, 2.86091034]])
model.predict(input)


# In[ ]:




