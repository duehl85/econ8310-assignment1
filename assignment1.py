#!/usr/bin/env python
# coding: utf-8

# In[79]:


import pandas as pd
from pygam import LinearGAM, s, f, l

#gather the data for the model and transform to date timne

data=pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv')
data['Timestamp']=pd.to_datetime(data['Timestamp'])
data['year']=data['Timestamp'].dt.year
data['month']=data['Timestamp'].dt.month
data['day']=data['Timestamp'].dt.weekday
data['hour']=data['Timestamp'].dt.hour

#gather the data to test the model

dataTest=pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv')
dataTest['Timestamp']=pd.to_datetime(dataTest['Timestamp'])
dataTest['year']=dataTest['Timestamp'].dt.year
dataTest['month']=dataTest['Timestamp'].dt.month
dataTest['day']=dataTest['Timestamp'].dt.weekday
dataTest['hour']=dataTest['Timestamp'].dt.hour

dataTest=dataTest[['month', 'day', 'hour']]

#create variables to build the model
x=data[['month', 'day', 'hour']]
y=data['trips']

#create the model
model=LinearGAM(s(0)+f(1)+s(2))
modelFit=model.gridsearch(x.values, y)

#make predictions
pred=modelFit.predict(dataTest.values)




# In[ ]:




