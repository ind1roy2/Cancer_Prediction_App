#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn


# In[2]:


df = pd.read_csv("E:/cell_samples.csv")


# In[3]:


df.head()


# In[4]:


df.Class.unique()


# In[5]:


df.shape


# In[6]:


type(df)


# In[7]:


df.Class.value_counts()


# In[8]:


df.describe()


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# In[11]:


df["BareNuc"].unique()


# In[12]:


df['BareNuc'].value_counts()


# In[13]:


df.BareNuc = df.BareNuc.replace('?','1')


# In[14]:


df["BareNuc"] = df["BareNuc"].astype('int64')


# In[15]:


df["BareNuc"].unique()


# In[16]:


df.dtypes


# In[17]:


df.duplicated().sum()


# In[18]:


df.shape


# In[19]:


df.loc[df.duplicated(),:]


# In[20]:


df = df.drop_duplicates(keep = "first")


# In[21]:


type(df)


# In[22]:


df.shape


# In[23]:


sns.boxplot(x = 'Clump',data=df)


# In[24]:


sns.boxplot(x = 'UnifShape',data=df)


# In[25]:


sns.boxplot(x = 'UnifSize',data=df)


# In[26]:


# MargAdh	SingEpiSize	BareNuc	BlandChrom	NormNucl	Mit
sns.boxplot(x = 'MargAdh',data=df)


# In[27]:


sns.boxplot(x = 'SingEpiSize',data=df)


# In[28]:


sns.boxplot(x = 'BareNuc',data=df)


# In[29]:


sns.boxplot(x = 'BlandChrom',data=df)


# In[30]:


sns.boxplot(x = 'NormNucl',data=df)


# In[31]:


sns.boxplot(x = 'Mit',data=df)


# In[32]:


f1 = df.MargAdh<8


# In[33]:


f2 = df.SingEpiSize<7


# In[34]:


f3 = df.BlandChrom<9


# In[35]:


f4 = df.NormNucl<8


# In[36]:


df = df[f1&f2&f3&f4]


# In[37]:


df.shape


# In[38]:


corr = df.corr()


# In[39]:


sns.heatmap(data = corr)


# In[40]:


df.columns


# In[41]:


from sklearn.model_selection import train_test_split


# In[49]:


X_train,X_test,y_train,y_test=train_test_split(df.drop(["ID","UnifShape","Mit","Class"],axis=1),df["Class"],test_size=0.2,random_state=2)


# In[50]:


X_train.shape


# In[51]:


X_test.shape


# In[52]:


y_train.shape


# In[53]:


y_test.shape


# In[54]:


from sklearn.preprocessing import StandardScaler


# In[55]:


s = StandardScaler()


# In[56]:


X_train = s.fit_transform(X_train)


# In[57]:


X_test = s.transform(X_test)


# In[58]:


type(X_train)


# In[59]:


type(X_test)


# In[61]:


y_train = y_train.to_numpy()


# In[62]:


y_test = y_test.to_numpy()


# In[63]:


from sklearn.linear_model import LogisticRegression 


# In[64]:


log = LogisticRegression()


# In[65]:


log.fit(X_train,y_train)


# In[66]:


p1 = log.predict(X_test)


# In[67]:


log.score(X_test,y_test)


# In[96]:


from sklearn.metrics import confusion_matrix,classification_report


# In[76]:


sns.heatmap(confusion_matrix(y_test,p1),annot = True)


# In[98]:


print(classification_report(y_test,p1))


# In[68]:


from sklearn import svm


# In[69]:


from sklearn.svm import SVC


# In[70]:


s1 = SVC(kernel = 'linear', gamma='auto',C=2)


# In[71]:


s1.fit(X_train,y_train)


# In[73]:


p2 = s1.predict(X_test)


# In[77]:


s1.score(X_test,y_test)


# In[78]:


sns.heatmap(confusion_matrix(y_test,p2),annot = True)


# In[100]:


print(classification_report(y_test,p2))


# In[79]:


from sklearn.tree import DecisionTreeClassifier


# In[80]:


dtc = DecisionTreeClassifier()


# In[81]:


dtc.fit(X_train,y_train)


# In[82]:


p3 = dtc.predict(X_test)


# In[83]:


dtc.score(X_test,y_test)


# In[84]:


sns.heatmap(confusion_matrix(y_test,p3),annot = True)


# In[102]:


print(classification_report(y_test,p3))


# In[85]:


from sklearn.ensemble import RandomForestClassifier


# In[86]:


rf = RandomForestClassifier()


# In[87]:


rf.fit(X_train,y_train)


# In[88]:


p4 = rf.predict(X_test)


# In[89]:


rf.score(X_test,y_test)


# In[90]:


sns.heatmap(confusion_matrix(y_test,p4),annot = True)


# In[103]:


print(classification_report(y_test,p4))


# In[91]:


from sklearn.neighbors import KNeighborsClassifier


# In[92]:


knn = KNeighborsClassifier()


# In[93]:


knn.fit(X_train,y_train)


# In[94]:


p5 = knn.predict(X_test)


# In[95]:


knn.score(X_test,y_test)


# In[138]:


sns.heatmap(data = cm,square = True,annot = True, cmap = "Blues")


# In[104]:


print(classification_report(y_test,p5))


# In[105]:


import pickle


# In[106]:


data = {'model':rf,'s':s}
with open('cancer.pkl','wb') as file:
    pickle.dump(data,file)


# In[107]:


with open('cancer.pkl', 'rb') as file:
    data = pickle.load(file)


# In[ ]:




