#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("C:\\Users\\gajendra singh\\OneDrive\\Desktop\\pandas\\sales_data.csv")


# In[3]:


df.head()


# Create customer segments based on purchasing behavior.
# 
# Calculate average purchase quantity and total spending for each segment.
# 
# Identify the preferred product categories for each segment.

# In[11]:


df['CustomerID'].value_counts()


# In[4]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Assuming 'Price' is already the original price without discount applied
df['Revenue'] = df['QuantitySold'] * df['Price'] * (1 - df['Discount'])

# Extract relevant features for clustering
X = df[['QuantitySold', 'Price', 'Discount', 'Revenue']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow method results to find the optimal number of clusters
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

# Based on the Elbow method, choose the optimal number of clusters and fit the model
optimal_clusters = 3  # Choose the number based on the elbow in the plot
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Display the customer segments
print("Customer Segments:")
print(df[['CustomerID', 'Cluster']])


# In[5]:


# Group by the 'Cluster' column
cluster_groups = df.groupby('Cluster')

# Calculate average purchase quantity and total spending for each segment
segment_metrics = cluster_groups.agg({
    'QuantitySold': 'mean',    # Average purchase quantity
    'Revenue': 'sum'           # Total spending
}).reset_index()

print("Customer Segment Metrics:")
print(segment_metrics)


# In[8]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data from the CSV file
df = pd.read_csv("C:\\Users\\gajendra singh\\OneDrive\\Desktop\\pandas\\sales_data.csv")

# Assuming 'Price' is already the original price without discount applied
df['Revenue'] = df['QuantitySold'] * df['Price'] * (1 - df['Discount'])

# Extract relevant features for clustering
X = df[['QuantitySold', 'Price', 'Discount', 'Revenue']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set the number of clusters
num_clusters = 5

# Fit the k-means model
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Group by the 'Cluster' column
cluster_groups = df.groupby('Cluster')

# Calculate average purchase quantity and total spending for each segment
segment_metrics = cluster_groups.agg({
    'QuantitySold': 'mean',    # Average purchase quantity
    'Revenue': 'sum'           # Total spending
}).reset_index()

print("Customer Segment Metrics:")
print(segment_metrics)


# In[9]:


category_distribution = df.groupby(['Cluster', 'Category']).size().reset_index(name='ProductCount')

# Identify the most preferred category for each segment
preferred_categories = category_distribution.loc[category_distribution.groupby('Cluster')['ProductCount'].idxmax()]

print("Preferred Product Categories for Each Segment:")
print(preferred_categories[['Cluster', 'Category', 'ProductCount']])


# In[ ]:




