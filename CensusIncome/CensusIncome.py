
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Analysis of Census Dataset using Standard ML Classification Algorithms (Stratification)
 
      1) Logistic Regression 
      2) Decision Tree 
      3) Random Forest
      4) SVM 
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.utils import resample
from pandas.plotting import scatter_matrix

from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# from sklearn import metrics, 
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc, accuracy_score,
                             precision_score, roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

from sklearn.model_selection import cross_val_score

from time import time 

from sklearn.linear_model import LogisticRegression                 # Logistic Regression
from sklearn.tree import DecisionTreeClassifier                     # Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier                 # Random Forest Classifier 
from sklearn.neighbors import KNeighborsClassifier                  # K Neighbours Classifier 
from sklearn.svm import SVC                                         # SVM


# Creating a Census Dataframe

Census = pd.read_csv(r'C:\Users\Bobby\Desktop\AUT Datasets\AdultIncome.csv')

# Explore the Data

Census.info()                          # 32561 (Rows) and 15 (Columns)  (7 Numerical Variables and 8 Categorical variables)


Census.describe()                    # Only gives the Numerical variables  6 Colunsn


#Code to check for shape of data
 
print ("Number of rows:  ", Census.shape)         # (32561, 15)
print ("Number of rows:  ", Census.shape[0])      # 32561 
print ("Number of Columns:  ", Census.shape[1])   # 15 


# Head 

Census.head(10)
Census.tail()


#Code to check for any missing values

Census.isnull().any()
Census.isnull().values.any()
Census.isnull().sum()  
Census.isna().any()

# Workclass, native.country, occupation have NaN values


# Rename the Target Variable Column to be a binary variable (<=50k = 0 (Class 0) and > 50k = 1 (Class 1))
# Census['income']=Census['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
# There were 24720 (76%) instances of <=50K (Class 0) and 7841 (24%) instances of >50K (Class 1) 

Census.income.replace(['<=50K', '>50K'], [0, 1], inplace=True)     
Census_count = Census['income'].value_counts()      
print(Census_count)      
    

# Plot of Class variable distribution

fig = sns.barplot(x = [0,1], y = Census_count, data = Census, color = 'blue')
plt.ylabel('Frequency of Class variable')
plt.xlabel ('Class variable')
plt.show(fig)


# Identify Numeric variables  (7 variables including Class (Income))
numeric_var = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week','income']

# Identify Categorical variables (8 variables))
categ_var = ['workclass','education','marital.status', 'occupation', 'relationship', 'race', 'sex', 'native']


#Converting Categorical variables into numerical variables

# Replace '?' in occupation with 0

Census['occupation'] = Census['occupation'].map({'?': 0, 'Farming-fishing': 1, 'Tech-support': 2, 
                                                       'Adm-clerical': 3, 'Handlers-cleaners': 4, 'Prof-specialty': 5,
                                                       'Machine-op-inspct': 6, 'Exec-managerial': 7, 
                                                       'Priv-house-serv': 8, 'Craft-repair': 9, 'Sales': 10, 
                                                       'Transport-moving': 11, 'Armed-Forces': 12, 'Other-service': 13, 
                                                       'Protective-serv': 14}).astype(int)


Census['sex'] = Census['sex'].map({'Male': 0, 'Female': 1}).astype(int)

Census['race'] = Census['race'].map({'Black': 0, 'Asian-Pac-Islander': 1, 'Other': 2, 'White': 3, 
                                             'Amer-Indian-Eskimo': 4}).astype(int)

Census["marital.status"] = Census["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
Census["marital.status"] = Census["marital.status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
Census["marital.status"] = Census["marital.status"].map({"Married":1, "Single":0})
Census["marital.status"] = Census["marital.status"].astype(int)

# Correlation matrix between variables

fig = Census.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(fig, vmax=.8,annot=True,cmap="PuBu", square=True);


# Plot Native Nation vs Income

fig = sns.barplot(x="native.country",y="income",data=Census)
fig = fig.set_ylabel("Income >50K")
plt.show(fig)

# Plot Sex vs Income

fig = sns.barplot(x="sex",y="income",data=Census)
fig = fig.set_ylabel("Income >50K")
plt.show(fig)

# Explore Relationship vs Income

fig = sns.factorplot(x="relationship",y="income",data=Census,kind="bar", size = 6.5 , palette =  'pastel')
fig.despine(left=True)
fig = fig.set_ylabels("Income >50K")
plt.show(fig)

# Explore Marital Status vs Income
fig = sns.factorplot(x="marital.status",y="income",data=Census,kind="bar", size = 6 , palette = "pastel")
fig.despine(left=True)
fig = fig.set_ylabels("Income >50K")
plt.show(fig)

# Explore Workclass vs Income

fig = sns.factorplot(x="workclass",y="income",data=Census,kind="bar", size = 6 , palette = "pastel")
fig.despine(left=True)
fig = fig.set_ylabels("Income >50K")
plt.show(fig)

# Explore Education vs Income

fig = sns.factorplot(x="education",y="income",data=Census,kind="bar", size = 10 , palette = "pastel")
fig.despine(left=True)
fig = fig.set_ylabels("Income >50K")
plt.show(fig)

# Explore Occupation vs Income

fig = sns.factorplot(x="occupation",y="income",data=Census,kind="bar", size = 10 , palette = "pastel")
fig.despine(left=True)
fig = fig.set_ylabels("Income >50K")
plt.show(fig)


# Fill missing Categorial Values

Census["workclass"] = Census["workclass"].fillna("X")
Census["native.country"] = Census["native.country"].fillna("United-States")


# Drop data

Census.drop(labels=["workclass","education","relationship","race","native.country"], axis = 1, inplace = True)
print(Census.head())

Census.head(100)

#################################################### Creating Train/Test Split Data ######################################

# Use Train/Test split with Random State Values and Stratified across both the classes (Fraud and Non-Fraud)
# Creating Input Features (X) and Target variable (y)

# Splitting into Feature and Target Variable

X = Census.loc[:, Census.columns != 'income']                 # [32561 rows x 9 columns]
y = Census.loc[:, Census.columns == 'income']                 # [32561 rows x 1 columns]

print(X)
print(y)

# Use Train/Test split with Random State Values
# Splitting the Data Set into 80% (Training) and 20% (Testing)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 123, stratify=y)

print(X_train.shape, X_test.shape)     # (26048, 9) (6513, 9)
print(y_train.shape, y_test.shape)     # (26048, 1) (6513, 1)

print(X_train)
print(len(X_train))


######################################################## Separating Minority and Majority Classes ######################################################################
#  Separating Majority and Minority Classes in Training Set before Balancing

## Calculating the length of Minority and Majority Class in the entire Census Dataframe
# There were 24720 (76%) instances of <=50K (Class 0) and 7841 (24%) instances of >50K (Class 1) 

Census_l50 = len(Census[Census['income']==0])                     # Length of Majority Class (<=50K)
Census_a50 = len(Census[Census['income']==1])                     # Length of Minority Class (>50K)

print(Census_l50)                            # 24720
print(Census_a50)                            # 7841


#  Indices of the Majority Class (<=50K) Class

Census_l50_index = Census[Census['income']==0].index
print(len(Census_l50_index))                 # 24720   

# Indices of the Minority Class (>50K) Class

Census_a50_index = Census[Census['income']==1].index
print(len(Census_a50_index))               # 7841



##################################### Balancing Data in the Training Set ######################################################################

## Calculating the length of Minority and Majority Class in the Training Set
## Undersample
# The number of Majority Class in X_train 

Census_Xtrain_l50_index=[i for i in X_train.index if i in Census_l50_index]
len(Census_Xtrain_l50_index)       #  19775 (i in Census_l50_index)

# The number of Minority Class in X_train 

Census_Xtrain_a50_index=[i for i in X_train.index if i not in Census_l50_index]
len(Census_Xtrain_a50_index)       # 6273 

# Downsampling Majority Class 

Census_Downsample = resample(Census_Xtrain_l50_index, replace = False, n_samples = 6273, random_state = 123)
len(Census_Downsample)            # 6273

# Concatenate the Minority Indexes and Majority Indexes
Census_Undersample_index =  np.concatenate([Census_Xtrain_a50_index,Census_Downsample])
len(Census_Undersample_index)                    # 12546



######################################################## Undersample Dataset ######################################################################

# Balanced Undersampled Data

Census_Undersample = Census.iloc[Census_Undersample_index]
print (Census_Undersample)                             # [12546 rows x 10 columns]

# Credit Card Undersample Count

Census_Undersample_count = pd.value_counts(Census_Undersample['income'])
print(Census_Undersample_count)
  
# CreditCard Undersample Bar Chart

fig = Census_Undersample_count.plot(kind='bar')
plt.ylabel('Frequency of UnderSampled Class variables')
plt.xlabel ('Class Variables')
plt.show(fig)


# Use Train/Test split with Random State Values across both the classes (Fraud and Non-Fraud)

x_under = Census_Undersample.loc[:, Census_Undersample.columns!='income']
y_under = Census_Undersample.loc[:, Census_Undersample.columns=='income']
 
x_under_train, x_under_test, y_under_train, y_under_test = train_test_split(x_under, y_under, test_size=0.20, random_state=123)

print(x_under_train.shape, x_under_test.shape)      # (10036, 9) (2510, 9)
print(y_under_train.shape, y_under_test.shape)      # (10036, 1) (2510, 1)


######################################################## Logistic Regression ######################################################################

# with Undersample dataset
lr_under = LogisticRegression()
lr_under.fit(x_under_train, y_under_train.values.ravel())

lr_under_predict = lr_under.predict(x_under_test)
lr_under_accuracy = accuracy_score(lr_under_predict, y_under_test)
lr_recall = recall_score(lr_under_predict, y_under_test)

print(lr_under_accuracy)                                        # 0.7661354581673306
print(lr_recall)                                                # 0.7443502824858758

print(classification_report(y_under_test, lr_under_predict))
print(confusion_matrix(y_under_test, lr_under_predict))


######################################################## Support Vector Machine (SVM) ######################################################################

# Building the initial Model using Gaussian Kernel

SVM = SVC(C=1, kernel = 'linear', random_state = 0)
SVM.fit(x_under_train, y_under_train.values.ravel())
SVM_Pred = SVM.predict(x_under_test)

print(accuracy_score(y_under_test,SVM_Pred))                # 0.9001088444928198
print(recall_score(y_under_test,SVM_Pred))                  # 0.9527027027027027
print(precision_score(y_under_test,SVM_Pred))               # 0.015088449531737774

# Print Classification and Confusion Matrix 
print(classification_report(y_under_test,SVM_Pred))
print(confusion_matrix(y_under_test,SVM_Pred))


################################################# Random Forest ######################################################################

RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 100)   # Change n_estimators to 100
RF.fit(x_under_train, y_under_train.values.ravel())
RF_Pred = RF.predict(x_under_test)

print(accuracy_score(y_under_test,RF_Pred))                # 0.800398406374502
print(recall_score(y_under_test,RF_Pred))                  # 0.7802971071149335
print(precision_score(y_under_test,RF_Pred))               # 0.819376026272578

# Print Classification and Confusion Matrix 
print(classification_report(y_under_test,RF_Pred))
print(confusion_matrix(y_under_test,RF_Pred))


########################################  Decision Tree Classifier ######################################################################

#  For Undersampled Data  

# RF = RandomForestClassifier(n_estimators = 30, max_depth = 10oob_score = True, random_state = 100)
# n_estimator:  The number of trees in the random forest classification
# Criterion: Loss fucntion used to measure the quality of split 
# Random State:  See used by the Random State Generator for randomising dataset

#  Splitting the Dataset into Training (80%) and Testing (20%)


# Decision Tree Classifier

DT = DecisionTreeClassifier()
DT.fit(x_under_train, y_under_train)
DT_Pred = DT.predict(x_under_test)

print(accuracy_score(y_under_test,DT_Pred))                # 0.7677290836653387
print(recall_score(y_under_test,DT_Pred))                  # 0.7677873338545739
print(precision_score(y_under_test,DT_Pred))               # 0.7744479495268138

# Print Classification and Confusion Matrix 
print(classification_report(y_under_test,DT_Pred))
print(confusion_matrix(y_under_test,DT_Pred))


##########################  Predict on Full Dataset using Undersampled Dataset #############################################


lr_pred_full = lr_under.predict(X_test)

# print(recall_score(y_test,y_pred_full, average = None)) 

print(accuracy_score(y_test,lr_pred_full))                    # 0.7386764931675112
print(recall_score(y_test,lr_pred_full))                      # 0.8144132653061225
print(precision_score(y_test,lr_pred_full))                   # 0.47507440476190477

# Print Classification and Confusion Matrix 
print(classification_report(y_test,lr_pred_full))
print(confusion_matrix(y_test,lr_pred_full))

##### SVM

SVM_pred_full = SVM.predict(X_test)

print(accuracy_score(y_test,SVM_pred_full))                    # 0.8854090798737557
print(recall_score(y_test,SVM_pred_full))                      # 0.7446120689655172
print(precision_score(y_test,SVM_pred_full))                   # 0.49427753934191704

# Print Classification and Confusion Matrix 
print(classification_report(y_test,SVM_pred_full))
print(confusion_matrix(y_test,SVM_pred_full))

##### RF

RF_predicted_full = RF.predict(X_test)

print(accuracy_score(y_test,RF_predicted_full))               # 0.8166743436204514
print(recall_score(y_test,RF_predicted_full))                 # 0.7933673469387755
print(precision_score(y_test,RF_predicted_full))              # 0.5884578997161779

# Print Classification and Confusion Matrix 
print(classification_report(y_test,RF_predicted_full))
print(confusion_matrix(y_test,RF_predicted_full))

##### Decision Tree

DT_predicted_full =  DT.predict(X_test)

print(accuracy_score(y_test,DT_predicted_full))                # 0.7701520036849379
print(recall_score(y_test,DT_predicted_full))                  # 0.7748724489795918
print(precision_score(y_test,DT_predicted_full))               # 0.5150487494701145

# Print Classification and Confusion Matrix 
print(classification_report(y_test,DT_predicted_full))
print(confusion_matrix(y_test,DT_predicted_full))


############################################## Upsample Dataset #######################################################################


# Upsampling Minority Class

Census_Upsample = resample(Census_Xtrain_a50_index, replace = True, n_samples = 19775, random_state = 123)
len(Census_Upsample)                    # 19775


# Concatenate the Minority Indexes and Majority Indexes
Census_Upsample_index =  np.concatenate([Census_Xtrain_l50_index,Census_Upsample])
len(Census_Upsample_index)               # 39550


# Balanced Upsampled Data
Census_Upsample = Census.iloc[Census_Upsample_index]
print (Census_Upsample)                                          # [39550 rows x 10 columns]

# Credit Card Oversample Count

Census_Upsample_count = pd.value_counts(Census_Upsample['income'])
print(Census_Upsample_count)
  
# CreditCard Undersample Bar Chart

fig = Census_Upsample_count.plot(kind='bar')
plt.ylabel('Frequency of Upsampled Class variables')
plt.xlabel ('Class Variables')
plt.show(fig)


# Use Train/Test split with Random State Values across both the classes (Fraud and Non-Fraud)

x_up = Census_Upsample.loc[:, Census_Upsample.columns!='income']
y_up = Census_Upsample.loc[:, Census_Upsample.columns=='income']
 
x_up_train, x_up_test, y_up_train, y_up_test = train_test_split(x_up, y_up, test_size=0.20, random_state=123)

print(x_up_train.shape, x_up_test.shape)      # (31640, 9) (7910, 9)
print(y_up_train.shape, y_up_test.shape)      # (31640, 1) (7910, 1)



######################################################## Logistic Regression ######################################################################

lr_up = LogisticRegression()
lr_up.fit(x_up_train, y_up_train.values.ravel())
lr_up_predict = lr_up.predict(x_up_test)

print(accuracy_score(y_up_test, lr_up_predict))         # 0.9527263934228025
print(recall_score(y_up_test, lr_up_predict))           # 0.9283089448784245
print(precision_score(y_up_test, lr_up_predict))        # 0.9758732361785797

print(classification_report(y_up_test, lr_up_predict))
print(confusion_matrix(y_up_test, lr_up_predict))

######################################################## Support Vector Machine (SVM) ######################################################################

# Building the initial Model using Gaussian Kernel

SVM_up = SVC(C=1, kernel = 'linear', random_state = 0)
SVM_up.fit(x_up_train, y_up_train.values.ravel())

SVM_Pred_up = SVM_up.predict(x_up_test)

print(accuracy_score(y_up_test, SVM_Pred_up))
print(recall_score(y_up_test, SVM_Pred_up))
print(precision_score(y_up_test, SVM_Pred_up)) 

print(classification_report(y_up_test, lr_up_predict))
print(confusion_matrix(y_up_test, lr_up_predict))


################################################# Random Forest ######################################################################


RF_up = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 100)   # Change n_estimators to 100


RF_up.fit(x_up_train, y_up_train.values.ravel())
RF_Pred_up = RF_up.predict(x_up_test)


print(accuracy_score(y_up_test, RF_Pred_up))
print(recall_score(y_up_test, RF_Pred_up))
print(precision_score(y_up_test, RF_Pred_up)) 

print(classification_report(y_up_test, RF_Pred_up))
print(confusion_matrix(y_up_test, RF_Pred_up))


########################################  Decision Tree Classifier ######################################################################

# RF = RandomForestClassifier(n_estimators = 30, max_depth = 10oob_score = True, random_state = 100)
# n_estimator:  The number of trees in the random forest classification
# Criterion: Loss fucntion used to measure the quality of split 
# Random State:  See used by the Random State Generator for randomising dataset

#  Splitting the Dataset into Training (80%) and Testing (20%)


# Decision Tree Classifier

DT_up = DecisionTreeClassifier()
DT_up.fit(x_up_train, y_up_train)
DT_Pred_up = DT.predict(x_up_test)


print(accuracy_score(y_up_test, DT_Pred_up))
print(recall_score(y_up_test, DT_Pred_up))
print(precision_score(y_up_test, DT_Pred_up)) 

print(classification_report(y_up_test, DT_Pred_up))
print(confusion_matrix(y_up_test, DT_Pred_up))



##########################  Predict on Full Dataset using Upsampled Dataset #############################################


lr_full_pred = lr_up.predict(X_test)

# print(recall_score(y_test,y_pred_full, average = None)) 

print(accuracy_score(y_test,lr_full_pred))               # 0.860403010439427
print(recall_score(y_test,lr_full_pred))                 # 0.8514030612244898
print(precision_score(y_test,lr_full_pred))              # 0.48580786026200873

print(classification_report(y_test,lr_full_pred))
print(confusion_matrix(y_test,lr_full_pred))

##### SVM

SVM_full_pred = SVM_up.predict(X_test)

print(accuracy_score(y_test,SVM_pred_full))                    # 0.8854090798737557
print(recall_score(y_test,SVM_pred_full))                      # 0.7446120689655172
print(precision_score(y_test,SVM_pred_full))                   # 0.49427753934191704

print(classification_report(y_test,SVM_pred_full))
print(confusion_matrix(y_test,SVM_pred_full))
 
##### RF

RF_full_pred = RF.predict(X_test)

print(accuracy_score(y_test,RF_full_pred))               # 0.8166743436204514
print(recall_score(y_test,RF_full_pred))                 # 0.7933673469387755
print(precision_score(y_test,RF_full_pred))              # 0.5884578997161779

print(classification_report(y_test,RF_full_pred))
print(confusion_matrix(y_test,RF_full_pred))

### Decision Tree

DT_full_pred =  DT.predict(X_test)

print(accuracy_score(y_test,DT_full_pred))                # 0.7701520036849379
print(recall_score(y_test,DT_full_pred))                  # 0.7748724489795918
print(precision_score(y_test,DT_full_pred))               # 0.5150487494701145

print(classification_report(y_test,DT_full_pred))
print(confusion_matrix(y_test,DT_full_pred))





