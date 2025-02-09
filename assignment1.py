#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import plotly.express as px
from pygam import LinearGAM, s, f, l
from plotly import tools
import plotly.offline as py
from plotly.subplots import make_subplots
import plotly.graph_objs as go

#read the data for fitting the model

data = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv')

#transform the data to be recognized as timestamp variables

data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['year']=data['Timestamp'].dt.year
data['month']=data['Timestamp'].dt.month
data['day']=data['Timestamp'].dt.weekday
data['hour']=data['Timestamp'].dt.hour

#read the data for testing the model

testData=pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv')

#transform the test data to be recongized as timestamp variables

testData['Timestamp'] = pd.to_datetime(testData['Timestamp'])
testData['year']=testData['Timestamp'].dt.year
testData['month']=testData['Timestamp'].dt.month
testData['day']=testData['Timestamp'].dt.weekday
testData['hour']=testData['Timestamp'].dt.hour

testData=testData[['month', 'day', 'hour']]


#create x & y variables based on the data
x=data[['month', 'day', 'hour']]
y=data['trips']

#create the model
model=LinearGAM(s(0)+f(1)+s(2))

#fit the model

modelFit=model.gridsearch(x.values, y)

#make predictions

pred=modelFit.predict(testData.values)


# In[68]:


pred


# In[55]:


#make the plots
#name the figures
titles=['month', 'day','hour']

#create the subplots in a single row grid

fig=make_subplots(rows=1, cols=3, subplot_titles=titles)
fig['layout'].update(height=500, width=1000, title='pyGAM', showlegend=False)

#loop over titles and create the corresponding figures

for i, title in enumerate(titles):
    XX=model.generate_X_grid(term=i)
    pdep, confi=model.partial_dependence(term=i, width=.95)
    trace = go.Scatter(x=XX[:,i], y=pdep, mode='lines', name='Effect')
    ci1=go.Scatter(x=XX[:,i], y=confi[:,0], line=dict(dash='dash', color='grey'), name='95% CI')
    ci2=go.Scatter(x=XX[:,i], y=confi[:,1], line=dict(dash='dash', color='grey'), name='95% CI')

    fig.append_trace(trace, 1, i+1)
    fig.append_trace(ci1, 1, i+1)
    fig.append_trace(ci2, 1, i+1)

py.iplot(fig)


# In[12]:


fig = px.line(data, x='Timestamp', y='trips',
              title='Timeseries of Trips',
              labels={'Timestamp': 'Timestamp',
                     'trips': 'Trips'})
fig.show()


# In[ ]:




