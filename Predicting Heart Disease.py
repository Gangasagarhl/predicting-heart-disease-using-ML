#!/usr/bin/env python
# coding: utf-8

# In[125]:


from Ipython.display import Image
import pydotplus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.externals.six import StringIO

get_ipython().run_line_magic('matplotlib', 'inline')


# # A) solution

# In[4]:


data=pd.read_csv("C:/Users/RAGHAVENDRA/Desktop/data sets from intellipaat/heart.csv")


# In[5]:


data.min()


# In[6]:


data.max()


# In[7]:


data.mean()


# In[8]:


data.corr()


# In[9]:


data.describe()


# In[10]:


data.info()


# # data visualisation

# In[11]:


import seaborn as sns


# In[84]:


sns.countplot(x=data.target,data=data,palette=["red","green"])
plt.title("[0] No heart disease  [1] Heart disease ")

plt.show()


# In[95]:


plt.figure(figsize=(18,10))
sns.countplot(x="age",hue="target",data=data,palette=["green","red"])
plt.legend(["No heart disease","Have heart disease"],loc="best")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()


# In[14]:


corr=data.corr()


# In[97]:


plt.figure(figsize=(18,10))
sns.heatmap(corr,annot=True)


# In[ ]:





# # c) solution Logistic Regression

# In[21]:


x=data.iloc[:,:-1]
y=data.iloc[:,-1]


# In[22]:


logre=LogisticRegression()


# In[166]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[167]:


logre.fit(x_train,y_train)


# In[168]:


y_pred=logre.predict(x_test)


# In[197]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[170]:


print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_train.shape)


# In[49]:


x_train.head()


# In[50]:


x_test.head()


# In[51]:


y_train.head()


# In[52]:


y_pred


# In[171]:


lr_cm=confusion_matrix(y_pred,y_test)
print("Confusion Matrix:\n\n",lr_cm)


# In[172]:


log_acc=accuracy_score(y_pred,y_test)
print("Accuracy_score:\n\n",log_acc)


# # decision Treee

# In[173]:


x_train,x_test,y_trsin,y_test=train_test_split(x,y,test_size=0.3,
                                               random_state=90)


# In[147]:


dectre= DecisionTreeClassifier()


# In[174]:


dectre.fit(x_train,y_train)


# In[175]:


y_predict=dectre.predict(x_test)


# In[176]:


dec_cf=confusion_matrix(y_pred,y_test)
print("Confusion Matrix:\n\n",dec_cf)


# In[152]:


dec_acc=accuracy_score(y_pred,y_test)
print("Accuracy score:\n\n",dec_acc)


# In[153]:


data=StringIO()


# In[154]:


export_graphviz(dectre,out_file=data)


# In[155]:


data.getvalue()


# In[156]:


graph=pydotplus.graph_from_dot_data(data.getvalue())


# In[157]:


Image(graph.create_png())


# # Random Forest

# In[73]:


from sklearn.ensemble import RandomForestClassifier


# In[177]:


ran=RandomForestClassifier(n_estimators=10)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,
                                               random_state=5)


# In[178]:


ran.fit(x_train,y_train)


# In[137]:


y_pred=ran.predict(x_test)


# In[179]:


ran_cf=confusion_matrix(y_pred,y_test)
ran_cf


# In[139]:


ran_acc=accuracy_score(y_pred,y_test)
ran_acc


# In[140]:


data=StringIO()
export_graphviz(dectre,out_file=data)
export_graphviz(ran.estimators_[0],out_file=data)


# In[141]:


graph=pydotplus.graph_from_dot_data(data.getvalue())


# In[142]:


Image(graph.create_png())


# In[158]:


scores_df=pd.DataFrame({
    
    "logistic_regression":[log_acc],
    "Decison_Tree":[dec_acc],
    "Random_forest":[ran_acc]
})


# In[159]:


scores_df


# In[161]:


scores_df.plot(kind="bar",figsize=(10,10))


# In[192]:


pd.DataFrame(lr_cm)


# In[194]:


conf={
    "logistic regression_confusion":lr_cm,
    "decisontTree_confusion":dec_cf,
    "randomforest_confusion":ran_cf,
}


# In[196]:


for l,matrix in conf.items():
    plt.title(l)
    sns.heatmap(matrix,annot=True)
    plt.show()


# In[199]:


dic=classification_report(y_pred,y_test,output_dict=True)


# In[203]:


ans=pd.DataFrame(dic).transpose()


# In[204]:


ans

