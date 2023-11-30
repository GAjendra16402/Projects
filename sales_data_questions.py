#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("C:\\Users\\gajendra singh\\OneDrive\\Desktop\\pandas\\sales_data.csv")


# In[3]:


df.head()


# # 1. Sales Performance Analysis:
# Problem Statement: Analyze the overall sales performance by calculating the total revenue, average profit margin, and the number of units sold. Identify the top-selling products and categories. Tasks:
# Calculate total revenue for each product and overall.
# Calculate the average profit margin for each product.
# Identify the top-selling products and categories.

# In[5]:


# Assuming 'Price' is already the original price without discount applied
df['Revenue'] = df['QuantitySold'] * df['Price'] * (1 - df['Discount'])

# Calculate the total revenue
total_revenue = df['Revenue'].sum()

print(f'Total Revenue: ${total_revenue:.2f}')


# In[6]:


df['ProfitMargin'] = (df['Revenue'] - df['Profit']) / df['Revenue']

# Calculate average profit margin
average_profit_margin = df['ProfitMargin'].mean()

# Calculate total number of units sold
total_units_sold = df['QuantitySold'].sum()

print(f'Average Profit Margin: {average_profit_margin:.2%}')
print(f'Total Units Sold: {total_units_sold}')


# In[8]:


product_sales = df.groupby(['ProductID', 'Category'])['Revenue'].sum().reset_index()

# Identify the top-selling products and categories
top_selling_products = product_sales.sort_values(by='Revenue', ascending=False)
top_selling_categories = df.groupby('Category')['Revenue'].sum().sort_values(ascending=False)

print("Top Selling Products:")
print(top_selling_products)

print("\nTop Selling Categories:")
print(top_selling_categories)


# In[9]:


df['Revenue'] = df['QuantitySold'] * df['Price'] * (1 - df['Discount'])

# Group by ProductID and sum up the revenue for each product
total_revenue_per_product = df.groupby('ProductID')['Revenue'].sum().reset_index()

print("Total Revenue Per Product:")
print(total_revenue_per_product)


# In[11]:


total_unit_sold = df.groupby('Category')['QuantitySold'].sum().reset_index()

print("Total Units Sold Per Category:")
print(total_unit_sold)


# In[ ]:




