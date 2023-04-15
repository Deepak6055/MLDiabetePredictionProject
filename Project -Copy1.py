#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns 


# In[6]:


data = pd.read_csv(r'C:\Users\deepa\Downloads\diabetes.csv')
data.head
data.info


# In[7]:


data.isnull().sum()


# In[8]:


data


# In[9]:


data['Outcome'].unique()


# In[22]:


data.corr()
sns.heatmap(data.corr(),annot=True,fmt="0.1f")


# In[43]:


data.hist()


# In[10]:


x = data.drop('Outcome',axis=1)
x


# In[12]:


y = data['Outcome']
y


# In[13]:


from sklearn.model_selection import train_test_split
xtrain,xt,ytrain,yt = train_test_split(x,y,test_size=0.2,random_state=30)


# In[14]:


xtrain.shape,xt.shape,ytrain.shape,yt.shape


# In[15]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
model = LR.fit(xtrain,ytrain)


# In[17]:


model.score(xt,yt)*100


# In[19]:


z = LR.predict(xt)
z


# In[34]:


import gradio as gr
import numpy as np


# In[40]:


def expense(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    x = np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    x = np.array(x).reshape(1,-1)
 #df = sc.transform(df)
    prediction = LR.predict(x)
    prediction = float(prediction)
    return prediction
    


# In[41]:


app = gr.Interface(fn=expense,
 inputs=[gr.inputs.Number(label="Pregnancies"),
 gr.inputs.Number(label="Glucose"),
 gr.inputs.Number(label="BloodPressure"),
 gr.inputs.Number(label="SkinThickness"),
 gr.inputs.Number(label="Insulin"),
 gr.inputs.Number(label="BMI"),
 gr.inputs.Number(label="DiabetesPedigreeF"),
 gr.inputs.Number(label="Age")
        ],
 outputs= "label",
 title="Developing an ML Model for Diabestes prediction"
 )


# In[42]:


app.launch(show_error=True)


# In[ ]:




