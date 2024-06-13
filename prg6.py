#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Define the dataset
data = [
 ['Low', 'Low', 2, 'No', 'Yes'],
 ['Low', 'Med', 4, 'Yes', 'Yes'],
 ['Low', 'Low', 4, 'No', 'Yes'],
 ['Low', 'Med', 4, 'No', 'No'],
 ['Low', 'High', 4, 'No', 'No'],
 ['Med', 'Med', 4, 'No', 'No'],
 ['Med', 'Med', 4, 'Yes', 'Yes'],
 ['Med', 'High', 2, 'Yes', 'No'],
 ['Med', 'High', 5, 'No', 'Yes'],
 ['High', 'Med', 4, 'Yes', 'Yes'],
 ['High', 'Med', 2, 'Yes', 'Yes'],
 ['High', 'High', 2, 'Yes', 'No'],
 ['High', 'High', 5, 'Yes', 'Yes']
]


# In[2]:


import pandas as pd
df = pd.DataFrame(data, columns=['Price', 'Maintenance', 'Capacity', 'Airbag', 'Profitable'])


# In[3]:


label_encoders = {}
for column in ['Price', 'Maintenance', 'Airbag', 'Profitable']:
 le = LabelEncoder()
 df[column] = le.fit_transform(df[column])
 label_encoders[column] = le


# In[5]:


X = df.drop(columns=['Profitable'])
y = df['Profitable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)


# In[6]:


y_pred = clf.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:




