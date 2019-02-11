"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Analysis of Credit Card Fraud using Standard ML Classification Algorithms (Stratification)
 
      1) Logistic Regression 
      2) Decision Tree 
      3) Random Forest
      4) SVM 
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.utils import resample

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
from sklearn.svm import SVC                                         # SVM


# Creating a CreditCard Dataframe

CreditCard = pd.read_csv(r'C:\Users\Bobby\Desktop\AUT Datasets\Creditcard.csv')  

# Explore the Data
CreditCard.head(10)

CreditCard.info()                          # 284807 (Rows) and 31 (Columns)

# Describe the Data

CreditCard.describe()

# Show the Columns 

CreditCard.columns

#Code to check for shape of data
 
print ("Number of rows:  ", CreditCard.shape)         # 284807 Instances (Rows)  and 31 Features (Columns) 
print ("Number of rows:  ", CreditCard.shape[0])      # Gives the number of (Rows) only  284807 Instances 
print ("Number of Columns:  ", CreditCard.shape[1])   # 31 Features (Columns)  

#Code to check for any missing values

CreditCard.isnull().any()
CreditCard.isnull().values.any()
CreditCard.isnull().sum()  

# Print Class values count of Target variable (Highly Imbalanced data)
# There were 492 instances of Fraud (Class 1) and 284315 instances of Non Fraudulent (Class 0) 

CreditCard_count = CreditCard['Class'].value_counts() 
print(CreditCard_count)

# Plot of Class variable distribution

fig = sns.barplot(x = [0, 1], y = CreditCard_count, data = CreditCard, color = 'blue')
plt.ylabel('Frequency of Class variable')
plt.xlabel ('Class variable')
plt.show(fig)

# Creating a new Column with Normalised Amount and removing the Amount and Time Column in DataFrame

CreditCard['Normalised Amount'] = StandardScaler().fit_transform(CreditCard['Amount'].values.reshape(-1,1))
CreditCard = CreditCard.drop(['Time', 'Amount'], axis=1)   
print (CreditCard)                     #  [284807 rows x 30 columns]


#################################################### Creating Train/Test Split Data ######################################

# Use Train/Test split with Random State Values and Stratified across both the classes (Fraud and Non-Fraud)
# Creating Input Features (X) and Target variable (y)

# Splitting into Features and Target Variable

X = CreditCard.loc[:, CreditCard.columns!= 'Class']
y = CreditCard.loc[:, CreditCard.columns== 'Class'] 
# Use Train/Test split with Random State Values
# Splitting the Data Set into 80% (Training) and 20% (Testing)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 123, stratify=y)

print(X_train.shape, X_test.shape)     # (227845, 29) (56962, 29)
print(y_train.shape, y_test.shape)     # (227845, 1) (56962, 1)

print(X_train)
print(len(X_train))


######################################################## Separating Minority and Majority Classes ######################################################################
#  Separating Majority and Minority Classes in Training Set before Balancing

## Calculating the length of Minority and Majority Class in the entire CreditCard Dataframe

CC_Non_Fraud = len(CreditCard[CreditCard['Class']==0])        # Length of Majority Class (Non-Fraud)
CC_Fraud = len(CreditCard[CreditCard['Class']==1])            # Length of Minority Class (Fraud)

print(CC_Non_Fraud)                                           # 284315
print(CC_Fraud)                                               # 492


#  Indices of the Majority (Non-Fraud) Class

CC_Non_Fraud_index = CreditCard[CreditCard['Class']==0].index
print(len(CC_Non_Fraud_index))                                    # 284315   

# Indices of the Minority (Fraud) Class

CC_Fraud_index = CreditCard[CreditCard['Class']==1].index
print(len(CC_Fraud_index))                                        # 492    

##################################### Balancing Data in the Training Set ######################################################################

## Calculating the length of Minority and Majority Class in the Training Set
## Undersample
# The number of Majority Class in X_train 

CC_Xtrain_NFraud_index=[i for i in X_train.index if i in CC_Non_Fraud_index]
len(CC_Xtrain_NFraud_index)       #  227451 (i in CC_Non_Fraud_index)

# The number of Minority Class in X_train 

CC_Xtrain_Fraud_index=[i for i in X_train.index if i not in CC_Non_Fraud_index]
len(CC_Xtrain_Fraud_index)       # 394 (i in CC_Fraud_index)

#. Randomly sample the majority indices with respect to the number of minority classes
CC_Downsample = resample(CC_Xtrain_NFraud_index, replace = False, n_samples = 394, random_state = 123)
len(CC_Downsample)

# Concatenate the Minority Indexes and Majority Indexes
CC_Undersample_index =  np.concatenate([CC_Xtrain_Fraud_index,CC_Downsample])
len(CC_Undersample_index)  

######################################################## Undersample Dataset ######################################################################

# Balanced Undersampled Data

CC_Undersample = CreditCard.iloc[CC_Undersample_index]
print (CC_Undersample)     

# Credit Card Undersample Count

CC_Undersample_count = pd.value_counts(CC_Undersample['Class'])
print(CC_Undersample_count)
  
# CreditCard Undersample Bar Chart

fig = CC_Undersample_count.plot(kind='bar')
plt.ylabel('Frequency of UnderSampled Class variables')
plt.xlabel ('Class Variables')
plt.show(fig)


# Use Train/Test split with Random State Values across both the classes (Fraud and Non-Fraud)

x_under = CC_Undersample.loc[:, CC_Undersample.columns!='Class']
y_under = CC_Undersample.loc[:, CC_Undersample.columns=='Class']
 
x_under_train, x_under_test, y_under_train, y_under_test = train_test_split(x_under, y_under, test_size=0.30, random_state=123)

print(x_under_train.shape, x_under_test.shape)      # (551, 29) (237, 29)
print(y_under_train.shape, y_under_test.shape)      # (551, 1) (237, 1)


######################################################## Logistic Regression ######################################################################

# with Undersample dataset

lr_under = LogisticRegression()
lr_under.fit(x_under_train, y_under_train.values.ravel())

lr_under_predict = lr_under.predict(x_under_test)

print(accuracy_score(lr_under_predict, y_under_test))           # 0.9620253164556962         
print(recall_score(lr_under_predict, y_under_test))             # 0.984  

# Print Classification and Confusion Matrix 
print(classification_report(y_under_test, lr_under_predict))
print(confusion_matrix(y_under_test, lr_under_predict))


######################################################## Support Vector Machine (SVM) ######################################################################

# Building the initial Model using Gaussian Kernel

SVM = SVC(C=1, kernel = 'linear', random_state = 0)
SVM.fit(x_under_train, y_under_train.values.ravel())
SVM_Pred = SVM.predict(x_under_test)

print(accuracy_score(y_under_test,SVM_Pred))                # 0.9620253164556962
print(recall_score(y_under_test,SVM_Pred))                  # 0.9461538461538461
print(precision_score(y_under_test,SVM_Pred))               # 0.984

# Print Classification and Confusion Matrix 

print(classification_report(y_under_test,SVM_Pred))
print(confusion_matrix(y_under_test,SVM_Pred))


################################################# Random Forest ######################################################################

# RF = RandomForestClassifier(n_estimators = 30, max_depth = 10oob_score = True, random_state = 100)
# n_estimator:  The number of trees in the random forest classification
# Criterion: Loss fucntion used to measure the quality of split 
# Random State:  See used by the Random State Generator for randomising dataset

RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 100)   # Change n_estimators to 100
RF.fit(x_under_train, y_under_train.values.ravel())
RF_Pred = RF.predict(x_under_test)

print(accuracy_score(y_under_test,RF_Pred))                # 0.9535864978902954
print(recall_score(y_under_test,RF_Pred))                  # 0.9307692307692308
print(precision_score(y_under_test,RF_Pred))               # 0.983739837398374

# Print Classification and Confusion Matrix 

print(classification_report(y_under_test,RF_Pred))
print(confusion_matrix(y_under_test,RF_Pred))

########################################  Decision Tree Classifier ######################################################################

# Decision Tree Classifier

DT = DecisionTreeClassifier()
DT.fit(x_under_train, y_under_train)
DT_Pred = DT.predict(x_under_test)

print(accuracy_score(y_under_test,DT_Pred))                # 0.9156118143459916
print(recall_score(y_under_test,DT_Pred))                  # 0.9384615384615385
print(precision_score(y_under_test,DT_Pred))               # 0.9104477611940298

# Print Classification and Confusion Matrix 

print(classification_report(y_under_test,DT_Pred))
print(confusion_matrix(y_under_test,DT_Pred))


##########################  Predict on Full Dataset using Undersampled Dataset #############################################


lr_pred_full = lr_under.predict(X_test)

# print(recall_score(y_test,y_pred_full, average = None)) 

print(accuracy_score(y_test,lr_pred_full))                    # 0.9646782065236473
print(recall_score(y_test,lr_pred_full))                      # 0.8775510204081632
print(precision_score(y_test,lr_pred_full))                   # 0.04122722914669223

# Print Classification and Confusion Matrix 

print(classification_report(y_test,lr_pred_full))
print(confusion_matrix(y_test,lr_pred_full))

##### SVM

SVM_pred_full = SVM.predict(X_test)

print(accuracy_score(y_test,SVM_pred_full))                    # 0.9695235420104631
print(recall_score(y_test,SVM_pred_full))                      # 0.8673469387755102
print(precision_score(y_test,SVM_pred_full))                   # 0.04701327433628318

# Print Classification and Confusion Matrix 

print(classification_report(y_test,SVM_pred_full))
print(confusion_matrix(y_test,SVM_pred_full))

##### RF

RF_predicted_full = RF.predict(X_test)

print(accuracy_score(y_test,RF_predicted_full))               # 0.9745970998209332
print(recall_score(y_test,RF_predicted_full))                 # 0.8877551020408163
print(precision_score(y_test,RF_predicted_full))              # 0.05712409717662508

# Print Classification and Confusion Matrix 

print(classification_report(y_test,RF_predicted_full))
print(confusion_matrix(y_test,RF_predicted_full))

##### Decision Tree

DT_predicted_full =  DT.predict(X_test)

print(accuracy_score(y_test,DT_predicted_full))                # 0.8934728415434852
print(recall_score(y_test,DT_predicted_full))                  # 0.8673469387755102
print(precision_score(y_test,DT_predicted_full))               # 0.013843648208469055

# Print Classification and Confusion Matrix 
print(classification_report(y_test,DT_predicted_full))
print(confusion_matrix(y_test,DT_predicted_full))

############################################## Upsample Dataset #######################################################################


# Upsampling Minority Class

CC_Upsample = resample(CC_Xtrain_Fraud_index, replace = True, n_samples = 227451, random_state = 123)
len(CC_Upsample)

# Concatenate the Minority Indexes and Majority Indexes

CC_Upsample_index =  np.concatenate([CC_Xtrain_NFraud_index,CC_Upsample])
len(CC_Upsample_index)  


# Balanced Upsampled Data

CC_Upsample = CreditCard.iloc[CC_Upsample_index]
print (CC_Upsample)                                                   #  [454902 rows x 30 columns]

# Credit Card Oversample Count

CC_Upsample_count = pd.value_counts(CC_Upsample['Class'])
print(CC_Upsample_count)
  
# CreditCard Undersample Bar Chart

fig = CC_Upsample_count.plot(kind='bar')
plt.ylabel('Frequency of Upsampled Class variables')
plt.xlabel ('Class Variables')
plt.show(fig)

# Use Train/Test split with Random State Values across both the classes (Fraud and Non-Fraud)

x_up = CC_Upsample.loc[:, CC_Upsample.columns!='Class']
y_up = CC_Upsample.loc[:, CC_Upsample.columns=='Class']
 
x_up_train, x_up_test, y_up_train, y_up_test = train_test_split(x_up, y_up, test_size=0.20, random_state=123)

print(x_up_train.shape, x_up_test.shape)      # (363921, 29) (90981, 29)
print(y_up_train.shape, y_up_test.shape)      # (363921, 1) (90981, 1)



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

SVM_up = SVC(C=1, kernel = 'rbf', random_state = 0)
SVM_up.fit(x_up_train, y_up_train.values.ravel())

SVM_Pred_up = SVM_up.predict(x_up_test)

print(accuracy_score(y_up_test, SVM_Pred_up))
print(recall_score(y_up_test, SVM_Pred_up))
print(precision_score(y_up_test, SVM_Pred_up)) 

print(classification_report(y_up_test, lr_up_predict))
print(confusion_matrix(y_up_test, lr_up_predict))



################################################# Random Forest ######################################################################

# RF = RandomForestClassifier(n_estimators = 30, max_depth = 10oob_score = True, random_state = 100)
# n_estimator:  The number of trees in the random forest classification
# Criterion: Loss fucntion used to measure the quality of split 
# Random State:  See used by the Random State Generator for randomising dataset

RF_up = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 100)   # Change n_estimators to 100


RF_up.fit(x_up_train, y_up_train.values.ravel())
RF_Pred_up = RF_up.predict(x_up_test)


print(accuracy_score(y_up_test, RF_Pred_up))              # 0.9999670260823689
print(recall_score(y_up_test, RF_Pred_up))                # 1.0
print(precision_score(y_up_test, RF_Pred_up))             # 0.9999339904946313

print(classification_report(y_up_test, RF_Pred_up))
print(confusion_matrix(y_up_test, RF_Pred_up))


########################################  Decision Tree Classifier ######################################################################

# Decision Tree Classifier

DT_up = DecisionTreeClassifier()
DT_up.fit(x_up_train, y_up_train)
DT_Pred_up = DT.predict(x_up_test)


print(accuracy_score(y_up_test, DT_Pred_up))                # 0.933931260373045
print(recall_score(y_up_test, DT_Pred_up))                  # 0.9782594344812411
print(precision_score(y_up_test, DT_Pred_up))               # 0.8984842360549717

print(classification_report(y_up_test, DT_Pred_up))
print(confusion_matrix(y_up_test, DT_Pred_up))



##########################  Predict on Full Dataset using Upsampled Dataset #############################################


lr_full_pred = lr_up.predict(X_test)

# print(recall_score(y_test,y_pred_full, average = None)) 

print(accuracy_score(y_test,lr_full_pred))               # 0.9771894713434688
print(recall_score(y_test,lr_full_pred))                 # 0.8673469387755102
print(precision_score(y_test,lr_full_pred))              # 0.06543494996150885

print(classification_report(y_test,lr_full_pred))
print(confusion_matrix(y_test,lr_full_pred))

##### SVM

SVM_full_pred = SVM_up.predict(X_test)

print(recall_score(y_test,SVM_pred_full))                      # 0.8851351351351351
print(accuracy_score(y_test,SVM_pred_full))                    # 0.9511136079023442
print(precision_score(y_test,SVM_pred_full))

print(classification_report(y_test,SVM_pred_full))
print(confusion_matrix(y_test,SVM_pred_full))
 
##### RF

RF_full_pred = RF.predict(X_test)

print(accuracy_score(y_test,RF_full_pred))               # 0.9745970998209332
print(recall_score(y_test,RF_full_pred))                 # 0.8877551020408163
print(precision_score(y_test,RF_full_pred))              # 0.05712409717662508

print(classification_report(y_test,RF_full_pred))
print(confusion_matrix(y_test,RF_full_pred))

### Decision Tree

DT_full_pred =  DT.predict(X_test)

print(accuracy_score(y_test,DT_full_pred))                # 0.892121063164917
print(recall_score(y_test,DT_full_pred))                  # 0.8571428571428571
print(precision_score(y_test,DT_full_pred))               # 0.013515687851971037

print(classification_report(y_test,DT_full_pred))
print(confusion_matrix(y_test,DT_full_pred))



