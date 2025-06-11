#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# # Data Cleaning

# In[2]:


import pandas as pd
data = pd.read_csv(r"D:\APPLICATION DEVELOPMENT\xxx_attack_detection\XSS_Dataset.csv")
data.head()
df = pd.DataFrame(data)
df


# In[3]:


import pandas as pd

df = pd.read_csv(r"D:\APPLICATION DEVELOPMENT\xxx_attack_detection\XSS_Dataset.csv")


# In[4]:


print("Original Dataset:")
print(df.head())


# In[5]:


missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)


# In[6]:


df = df.drop_duplicates()
df = df.dropna()
print("\nCleaned Dataset:")
print(df.head())


# In[7]:


import pandas as pd
from imblearn.over_sampling import RandomOverSampler
df = pd.read_csv(r"D:\APPLICATION DEVELOPMENT\xxx_attack_detection\XSS_Dataset.csv")

# Display the class distribution before balancing
print("Class Distribution before Balancing:")
print(df['Class'].value_counts())

# Separate features (X) and labels (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Use RandomOverSampler to balance the dataset
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Display the class distribution after balancing
print("\nClass Distribution after Balancing:")
print(pd.Series(y_resampled).value_counts())

# Now X_resampled and y_resampled can be used for training the machine learning model


# ## Preparing Features and Labels

# In[8]:


X = df.drop('Class', axis=1)
y = df['Class']


# ## Splitting the dataset

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Training Machine Learning Models

# ## Random Forest (RF)

# In[10]:


rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


# In[11]:


rf_predictions = rf_model.predict(X_test)


# In[12]:


print("Random Forest Model:")
print(f"Accuracy: {accuracy_score(y_test, rf_predictions)}")
print("Classification Report:")
print(classification_report(y_test, rf_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))


# ## Logistic Regression

# In[13]:


lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)


# In[14]:


lr_predictions = lr_model.predict(X_test)


# In[15]:


print("\nLogistic Regression Model:")
print(f"Accuracy: {accuracy_score(y_test, lr_predictions)}")
print("Classification Report:")
print(classification_report(y_test, lr_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))


# ## k-Nearest Neighbors Classifier

# In[16]:


knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)


# In[17]:


knn_predictions = knn_model.predict(X_test)


# In[18]:


print("\nk-Nearest Neighbors Model:")
print(f"Accuracy: {accuracy_score(y_test, knn_predictions)}")
print("Classification Report:")
print(classification_report(y_test, knn_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, knn_predictions))


# ## Support Vector Machine

# In[19]:


svm_model = SVC()
svm_model.fit(X_train, y_train)

+
# In[20]:


svm_predictions = svm_model.predict(X_test)


# In[21]:


print("\nSupport Vector Machine Model:")
print(f"Accuracy: {accuracy_score(y_test, svm_predictions)}")
print("Classification Report:")
print(classification_report(y_test, svm_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_predictions))


# In[ ]:





# In[ ]:




