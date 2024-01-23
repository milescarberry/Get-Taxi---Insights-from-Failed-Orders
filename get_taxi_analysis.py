#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns


from plotly import express as exp, graph_objects as go, io as pio


from plotly.subplots import make_subplots

from pandas_utils.pandas_utils_2 import *

# import ipywidgets as widgets

# from IPython.display import display

import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

pio.templates.default = 'ggplot2'

sns.set_style('darkgrid')

sns.set_context('paper', font_scale = 1.4)


    


# In[2]:


offers_df = pd.read_csv("./datasets/data_offers.csv")

orders_df = pd.read_csv("./datasets/data_orders.csv")


offers_df.shape, orders_df.shape


# In[3]:


offers_df.head(2)


# In[4]:


orders_df.head(2)


# In[5]:


orders_df = pd.merge(orders_df, offers_df, on = 'order_gk', how = 'left')


# In[6]:


orders_df = orders_df[
    ['order_datetime', 'origin_longitude', 'origin_latitude', 'm_order_eta', 'order_gk', 'offer_id', 'order_status_key', 'is_driver_assigned_key', 'cancellations_time_in_seconds']
]


# In[7]:


orders_df = orders_df.rename({"cancellations_time_in_seconds": "cancellation_time_in_seconds"}, axis = 1)


# In[8]:


show_nan(orders_df)


# In[9]:


orders_df['order_status'] = orders_df.order_status_key.apply(lambda x: "cancelled_by_client" if x == 4 else "cancelled_by_system")


# In[10]:


pd.DataFrame(orders_df[['order_status_key', 'order_status']].value_counts() / len(orders_df))


# In[11]:


# Offers are not applied on 9% of the total cancelled rides.


# In[12]:


# Cancellation time in seconds --> Cancellation time in seconds by client.


# In[13]:


orders_df['m_order_eta_is_null'] = orders_df.m_order_eta.apply(lambda x: 1 if 'nan' in str(x).lower() else 0)


# In[14]:


# orders_df.is_driver_assigned_key = orders_df.is_driver_assigned_key.apply(lambda x: str(x))


# In[15]:


pd.DataFrame(orders_df[['m_order_eta_is_null', 'is_driver_assigned_key']].value_counts()) / len(orders_df)


# In[16]:


# m_order_eta only when a driver is assigned.


# In[17]:


orders_df = orders_df.drop(columns = ['order_status_key'])


# In[18]:


orders_df.columns.values.tolist()


# In[19]:


pd.DataFrame(orders_df[['m_order_eta_is_null', 'order_status']].value_counts() / len(orders_df))


# In[20]:


stacked_bar_chart_ci_2(orders_df, 'm_order_eta_is_null', 'order_status')


# In[21]:


# When a driver is assigned, most of the ride cancellations are done by the client.


# In[22]:


orders_df[['cancellation_time_in_seconds', "m_order_eta"]].corr()


# In[23]:


orders_df.columns.values.tolist()


# In[24]:


orders_df['is_no_offer'] = orders_df['offer_id'].apply(lambda x: '1' if 'nan' in str(x).lower() else '0')


# In[25]:


pd.DataFrame(orders_df[['order_status', 'is_no_offer']].value_counts() / len(orders_df))


# In[26]:


stacked_bar_chart_ci_2(orders_df, 'order_status', 'is_no_offer')


# In[27]:


orders_df.columns.values.tolist()


# In[28]:


'''

Filters:

order_datetime (slider)

is_no_offer,

is_driver_assigned_key,

order_status


Calcs:

m_order_eta kdeplot

cancellation_time_in_seconds kdeplot



'''


# In[29]:


distinct_hours = orders_df.order_datetime.sort_values(ascending = True).apply(lambda x: str(x).split(":")[0]).unique()


distinct_hours = list(distinct_hours)

distinct_hours.sort()


hour_indices = [i for i in range(len(distinct_hours))]



# In[ ]:




