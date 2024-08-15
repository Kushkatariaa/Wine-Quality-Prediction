#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from warnings import filterwarnings
filterwarnings(action='ignore')


# In[9]:


wine=pd.read_csv(r'D:\Downloads\archive\WineQT.csv')
print(wine)


# In[10]:


wine.head()


# In[12]:


wine.tail()


# In[11]:


wine.describe(include='all')


# In[13]:


wine.groupby('quality').mean()


# In[14]:


sns.countplot(wine['quality'])
plt.show()


# In[15]:


sns.countplot(wine['pH'])
plt.show()


# In[16]:


sns.distplot(wine['alcohol'])


# In[17]:


sns.kdeplot(wine.query('quality > 2').quality)


# In[18]:


wine.hist(figsize=(10,10),bins=50)
plt.show()


# In[44]:


le = LabelEncoder()
wine['quality'] = le.fit_transform(wine['quality'])
wine.tail()


# In[22]:


wine['quality'].value_counts()


# In[46]:


ranges=(0,3,5) 
groups=['bad','good']
wine['quality']=pd.cut(wine['quality'],bins=ranges,labels=groups)


# In[27]:


good_quality = wine[wine['quality']==1]
bad_quality = wine[wine['quality']==0]

bad_quality = bad_quality.sample(frac=1)
bad_quality = bad_quality[:6]

new_df = pd.concat([good_quality,bad_quality])
new_df = new_df.sample(frac=1)
new_df


# In[24]:


new_df['quality'].value_counts()


# In[ ]:


wine['goodquality'] = [1 if x >= 3 else 0 for x in wine['quality']]
X = wine.drop(['quality','goodquality'], axis = 1)
Y = wine['goodquality']


# In[ ]:




