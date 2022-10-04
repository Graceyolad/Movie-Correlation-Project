#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libraries

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None


# In[3]:


#importing the data
df = pd.read_csv('movies.csv')


# In[4]:


#taking a look at our data
df.head(10)


# Data Exploration

# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.info()


# In[8]:


#checking if there are missing data
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{}-{}%'.format(col,pct_missing))


# In[9]:


#changing column data types
df['budget'] = df['budget'].fillna(0).astype('int64')
df['gross'] = df['gross'].fillna(0).astype('int64')


# In[10]:


df


# In[11]:


#creating correct year column due to the irregularities between released and year columns
df['yearcorrect'] = df['released'].str.extract(pat = '([0-9]{4})').astype(str)
df


# In[12]:


df.sort_values(by = ['gross'], inplace = False, ascending = False )


# In[13]:


pd.set_option('display.max_rows', None)


# In[15]:


#removing duplicates if available
df.drop_duplicates()


# In[17]:


plt.scatter(x = df['budget'], y = df['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel('Budget')
plt.ylabel('Gross Earnings')
plt.show()


# In[19]:


#budget vs gross plot using seaborn
sns.regplot(x = 'budget', y= 'gross', data =df, scatter_kws = {'color' : 'red'}, line_kws = {'color' : 'blue'})


# Correlation

# In[20]:


df.corr()


# In[22]:


#High correlation between budget and gross
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot = True)
plt.title('Correlation Matrix of Numerical Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[24]:


#looking at company
df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes

df_numerized


# In[25]:


correlation_matrix = df_numerized.corr()
sns.heatmap(correlation_matrix, annot = True)
plt.title('Correlation Matrix of Categorical Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[26]:


df_numerized.corr()


# In[28]:


correlation_mat = df_numerized.corr()
corr_pairs = correlation_mat.unstack()
corr_pairs


# In[29]:


sorted_pairs = corr_pairs.sort_values()
sorted_pairs


# In[30]:


high_corr = sorted_pairs[(sorted_pairs) > 0.5]
high_corr


# In[ ]:


#votes and budget have the highest correlation to gross earnings which makes total sense!

