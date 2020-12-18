#!/usr/bin/env python
# coding: utf-8

# # Loan Prediction Problem

# Find out the person is eligible to acquire loan based on their qualifications, employment, earning, dependent, their dependent’s income, credit history, their loan amount, and loan term. Create a machine learning model to generate loan approval from person’s information.

# Dataset Source: https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset

# In[5]:


from IPython.display import Image
Image("data/AboutDataset.JPG")


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


ds_loan = pd.read_csv('data/train.csv')
ds_loan_test = pd.read_csv('data/test.csv')


# # EDA on Train Dataset

# In[8]:


ds_loan


# In[9]:


ds_loan.describe()


# In[10]:


ds_loan.info()


# Check null values and clean train dataset

# In[11]:


ds_loan.isnull().sum()


# Except for LoanAmount column, every other null containing attributes have two unique values so by using mode function I'll fill missing values.
# 
# For LoanAmount attribute I'll use median function to resolve null value problem

# In[12]:


ds_loan['LoanAmount'].value_counts()


# In[13]:


ds_loan['Gender'].fillna(ds_loan['Gender'].mode()[0], inplace = True)
ds_loan['Married'].fillna(ds_loan['Married'].mode()[0], inplace = True)
ds_loan['Dependents'].fillna(ds_loan['Dependents'].mode()[0], inplace = True)
ds_loan['Self_Employed'].fillna(ds_loan['Self_Employed'].mode()[0], inplace = True)
ds_loan['LoanAmount'].fillna(ds_loan['LoanAmount'].median(), inplace = True)
ds_loan['Loan_Amount_Term'].fillna(ds_loan['Loan_Amount_Term'].mode()[0], inplace = True)
ds_loan['Credit_History'].fillna(ds_loan['Credit_History'].mode()[0], inplace = True)


# Training dataset after resolving missing values

# In[14]:


ds_loan


# In[15]:


ds_loan.isnull().sum()


# Training dataset dont have any duplicated values in it

# In[16]:


ds_loan.duplicated().sum()


# Now I'll apply EDA and data cleaning on test dataset

# # EDA on Test Dataset

# In[17]:


ds_loan_test


# In[18]:


ds_loan_test.describe()


# In[19]:


ds_loan_test.info()


# Check null values and clean train dataset test dataset

# In[20]:


ds_loan_test.isnull().sum()


# In[21]:


ds_loan_test['LoanAmount'].value_counts()


# In[22]:


ds_loan_test['Gender'].fillna(ds_loan_test['Gender'].mode()[0], inplace = True)
ds_loan_test['Dependents'].fillna(ds_loan_test['Dependents'].mode()[0], inplace = True)
ds_loan_test['Self_Employed'].fillna(ds_loan_test['Self_Employed'].mode()[0], inplace = True)
ds_loan_test['LoanAmount'].fillna(ds_loan_test['LoanAmount'].median(), inplace = True)
ds_loan_test['Loan_Amount_Term'].fillna(ds_loan_test['Loan_Amount_Term'].mode()[0], inplace = True)
ds_loan_test['Credit_History'].fillna(ds_loan_test['Credit_History'].mode()[0], inplace = True)


# In[23]:


ds_loan_test


# In[24]:


ds_loan_test.isnull().sum()


# In[25]:


ds_loan_test.duplicated().sum()


# In[26]:


ds_loan.drop('Loan_ID',axis=1,inplace=True)
ds_loan_test.drop('Loan_ID',axis=1,inplace=True)
#checking the new shapes
print(f"training shape (row, col): {ds_loan.shape}\n\ntesting shape (row, col): {ds_loan_test.shape}")


# # Data Visualization

# Creating collumns of all numerical and categorical columns, and dataset as well

# In[27]:


#list of all the numeric columns
num = ds_loan.select_dtypes('number').columns.to_list()
#list of all the categoric columns
cat = ds_loan.select_dtypes('object').columns.to_list()

#numeric dataset
ds_loan_num =  ds_loan[num]
#categoric dataset
ds_loan_cat = ds_loan[cat]


# Comparing Loan status output, where loan acceptance racio is higher then rejection

# In[28]:


sns.countplot(ds_loan['Loan_Status'])


# Histogram of all numerical data. We have got some interesting insights, they are mentioned below:
# 
# - in ApplicantIncome candidate have more income under 10000
# - in CoappicantIncome have more income bar around 5000
# - Candiate got more loan between 100 to 150
# - They all applied for long term loan payment
# - Most applicate have good credit history

# In[29]:


for a in ds_loan_num:
    plt.hist(ds_loan_num[a])
    plt.title(a)
    plt.show()


# Bar comparition of all categorical type attributes with loan status, and their insights are mentioned below:
# 
# - It seems that male have applied more for loan and also got more rejected too
# - Married couple have applied for loan than singles
# - Less depended in family got good acceptance rate
# - Loan company trust more on Gradate candidate and not self-employed(they want steady salary recored)
# - There is not much difference with where candidate live, but suburban have got good acceptance rate.

# In[30]:


for i in cat[:-1]: 
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    sns.countplot(x=i ,hue='Loan_Status', data=ds_loan )
    plt.xlabel(i, fontsize=14)


# Converting Object data into numerical data to better understand data relations.

# In[31]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
c = ['Gender', 'Married', 'Education', 
        'Self_Employed', 'Property_Area', 
        'Loan_Status', 'Dependents']
for i in c:
    ds_loan[i] = LE.fit_transform(ds_loan[i])

ds_loan


# Correlation of all columns after changing categorical data into numerical

# In[32]:


corr = ds_loan.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(corr, annot=True, cmap='cubehelix_r');


# In[33]:


X = ds_loan.drop('Loan_Status',1)
y = ds_loan['Loan_Status']


# In[34]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[35]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Model Evaluation

# I'll apply logistic regression, beacuse it gives better results with numeric and binary data.

# In[36]:


from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# Train KNeighborsClassifier Model
KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
KNN_Classifier.fit(x_train, y_train); 

# Train LogisticRegression Model
LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)
LGR_Classifier.fit(x_train, y_train);
            
# Train Decision Tree Model
DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
DTC_Classifier.fit(x_train, y_train)


# In[37]:


models = []

models.append(('Decision Tree Classifier', DTC_Classifier))
models.append(('KNeighborsClassifier', KNN_Classifier))
models.append(('LogisticRegression', LGR_Classifier))

for i, v in models:
    scores = cross_val_score(v, x_train, y_train, cv=10)
    accuracy = metrics.accuracy_score(y_train, v.predict(x_train))
    confusion_matrix = metrics.confusion_matrix(y_train, v.predict(x_train))
    classification = metrics.classification_report(y_train, v.predict(x_train))
    print()
    print('============================== {} Model Evaluation =============================='.format(i))
    print()
    print ("Cross Validation Mean Score:" "\n", scores.mean())
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print()


# From above result we can see that we have used 3 algotithms which are good for handling numerical and boolean values. Here Decision Tree gave overfitting result and Logistic Regression's output is better than K-Neighbours classifier algorithm. So I will try to improve Logestic Regression by using feature.

# In[38]:


LGR_Classifier.coef_


# In[39]:


feature_dict = dict(zip(ds_loan.columns, list(LGR_Classifier.coef_[0])))
features = pd.DataFrame(feature_dict, index=[0])
features.plot.bar(figsize=(10, 6));


# In[40]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# In[41]:


X1 = X[['Credit_History', 'ApplicantIncome', 'CoapplicantIncome', 'Married', 'Property_Area', 'Dependents', 'LoanAmount']]


# In[42]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X1, y, test_size = 0.25)


# In[43]:


# Train KNeighborsClassifier Model
KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
KNN_Classifier.fit(x_train, y_train); 

# Train LogisticRegression Model
LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)
LGR_Classifier.fit(x_train, y_train);


# In[44]:


models = []

models.append(('KNeighborsClassifier', KNN_Classifier))
models.append(('LogisticRegression', LGR_Classifier))

for i, v in models:
    scores = cross_val_score(v, x_train, y_train, cv=10)
    accuracy = metrics.accuracy_score(y_train, v.predict(x_train))
    confusion_matrix = metrics.confusion_matrix(y_train, v.predict(x_train))
    classification = metrics.classification_report(y_train, v.predict(x_train))
    print()
    print('============================== {} Model Evaluation =============================='.format(i))
    print()
    print ("Cross Validation Mean Score:" "\n", scores.mean())
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print()


# Using feature we are getting little improvement or slight decrement in accuracy, so I'll use all attributes to train the model

# Final Approach

# In[45]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[46]:


LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)
LGR_Classifier.fit(x_train, y_train);


# In[47]:


models = []

models.append(('LogisticRegression', LGR_Classifier))

for i, v in models:
    scores = cross_val_score(v, x_train, y_train, cv=10)
    accuracy = metrics.accuracy_score(y_train, v.predict(x_train))
    confusion_matrix = metrics.confusion_matrix(y_train, v.predict(x_train))
    classification = metrics.classification_report(y_train, v.predict(x_train))
    print()
    print('============================== {} Model Evaluation =============================='.format(i))
    print()
    print ("Cross Validation Mean Score:" "\n", scores.mean())
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print()


# In[48]:


for i, v in models:
    accuracy = metrics.accuracy_score(y_test, v.predict(x_test))
    confusion_matrix = metrics.confusion_matrix(y_test, v.predict(x_test))
    classification = metrics.classification_report(y_test, v.predict(x_test))
    print()
    print('============================== {} Model Test Results =============================='.format(i))
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print() 


# In[49]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
c = ['Gender', 'Married', 'Education', 
        'Self_Employed', 'Property_Area', 
        'Dependents']
for i in c:
    ds_loan_test[i] = LE.fit_transform(ds_loan_test[i])


# In[50]:


pred_log = LGR_Classifier.predict(ds_loan_test)


# In[51]:


pred_df = pd.DataFrame(pred_log, columns = ['class']) 
test_output = pd.concat([ds_loan_test, pred_df],axis=1)


# In[52]:


test_output


# In[53]:


sns.countplot(data = test_output, x = 'class')


# # Model Deployment steps

# In[54]:


import pickle


# In[55]:


filename = 'model.pkl'
pickle.dump(LGR_Classifier, open('model.pkl', 'wb'))


# In[56]:


model = pickle.load(open('model.pkl', 'rb'))

