#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_d = pd.read_csv('C:/Users/Asus/OneDrive/Documents/Training Data.csv')
test = pd.read_csv('C:/Users/Asus/OneDrive/Documents/Test Data.csv')
target_test = pd.read_csv('C:/Users/Asus/OneDrive/Documents/Sample Prediction Dataset.csv')


# In[3]:


train_d.drop('Id',axis=1,inplace=True)
train_d.head()


# In[4]:


test.head()


# In[5]:


target_test.head()


# In[6]:


test['Risk_Flag'] = target_test['risk_flag']
test.head()


# In[7]:


test.drop('ID',axis=1,inplace=True)
test.head()


# In[8]:


full_data = pd.concat([train_d,test],axis = 0)
full_data.shape


# In[9]:


full_data.head()
full_data.drop('CITY', axis=1, inplace=True)
cols_to_encode = ['Married/Single','House_Ownership', 'Car_Ownership','Profession','STATE']


# In[10]:


def one_hot_encode(data):
    # Step 1: Identify unique categories
    unique_categories = list(set(data))
    
    # Step 2: Create one-hot encoding
    encoded_data = []
    for value in data:
        encoding = [0] * len(unique_categories)  # Initialize encoding with zeros
        category_index = unique_categories.index(value)  # Find index of category
        encoding[category_index] = 1  # Set the corresponding position to one
        encoded_data.append(encoding)
    
    return encoded_data, unique_categories


# In[11]:


extra = pd.DataFrame()
for col in cols_to_encode:
    
    full_data[col] = full_data[col].str.replace('_', ' ')
    data = full_data[col]
    encoded_data, unique_categories = one_hot_encode(data)

    df = pd.DataFrame(encoded_data, columns=unique_categories)
    df = df.astype(int)
    
    extra = pd.concat([extra, df], axis=1)
    


# In[12]:


extra.head()


# In[14]:


# cols_to_encode = ['Married/Single','House_Ownership', 'Car_Ownership','Profession','CITY','STATE']
# dummies = pd.get_dummies(full_data[cols_to_encode], drop_first=True)
# dummies = dummies.astype(int)
# dummies.shape


# In[15]:


# dummies.head()


# In[13]:


full_data.drop(cols_to_encode, axis=1, inplace=True)
full_data.head()


# In[14]:


scale = MinMaxScaler()
scalled = scale.fit_transform(full_data.drop('Risk_Flag',axis=1))


# In[15]:


i = 0
for col in full_data.columns[:-1]:
    full_data[col] = scalled[:,i]
    i += 1


# In[16]:


full_data.head()


# In[17]:


print(full_data.index.duplicated().any())
print(extra.index.duplicated().any())


# In[18]:


full_data = full_data.reset_index(drop=True)  # Reset index to make it unique
extra = extra.reset_index(drop=True)  # Reset index to make it unique


# In[19]:


print(full_data.index.duplicated().any())
print(extra.index.duplicated().any())


# In[20]:


full_data = pd.concat([full_data, extra], axis=1)
full_data.head()


# In[21]:


full_data['Risk_Flag'].value_counts()


# In[22]:


class0 = full_data[full_data['Risk_Flag'] == 0].sample(34589)


# In[23]:


class1 = full_data[full_data['Risk_Flag'] == 1].sample(34589)


# In[24]:


full_data2 = pd.concat([class0,class1],axis = 0)
full_data2.shape


# In[25]:


x,y = full_data2.drop('Risk_Flag',axis = 1),full_data2['Risk_Flag']
x.shape, y.shape


# In[26]:


X_train, X_test, Y_train, Y_test = train_test_split(x,y,random_state=1)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# In[27]:


model = [LogisticRegression]


# In[28]:


model = keras.Sequential([
    keras.layers.Dense(92, input_shape=(92,), activation='relu'),
#     keras.layers.Dense(60, activation='relu'),
#     keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# In[29]:


model.fit(X_train, Y_train, epochs=150,batch_size=1024)


# In[37]:


model.evaluate(X_test, Y_test)


# In[ ]:




