#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np


# In[10]:


data_frame=pd.read_csv('final_data2.csv')


# In[11]:


data_frame.head()


# In[12]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(data_frame,test_size=0.25,random_state=42)


# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.9,random_state=42)
for train_indices,test_indices in split.split(data_frame,data_frame['crop_type']):
    strat_train_set=data_frame.loc[train_indices]
    strat_test_set=data_frame.loc[test_indices]
 


# In[15]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
myPipeline=Pipeline([
    
    ('std_scaler',StandardScaler())
])


# In[16]:


data_frame=strat_train_set.drop("Target",axis=1)
data_frame_labels=strat_train_set["Target"]


# In[17]:


data_frame_transform=myPipeline.fit_transform(data_frame)


# In[18]:


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(max_leaf_nodes=200,n_estimators=130,min_samples_split=3,random_state=3)
model.fit(data_frame_transform,data_frame_labels)


# In[21]:


# from sklearn.model_selection import cross_val_score
# scores=cross_val_score(model,data_frame_transform,data_frame_labels,scoring="neg_mean_squared_error",cv=10)
# rmse_scores=np.sqrt(-scores)
# print(rmse_scores)


# ## Predicting 
# 

# In[27]:


from joblib import dump, load
dump(model, 'model.pkl') 

