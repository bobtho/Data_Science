
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Analysis of PIMA Indian Diabetes using Standard ML Classification Algorithms (Stratification)
 
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
from sklearn.neighbors import KNeighborsClassifier                  # K Neighbours Classifier 
from sklearn.svm import SVC                                         # SVM


# Creating a PIMA Dataframe

#PIMA = pd.read_csv(r'H:\AUT Datasets\Creditcard.csv')                              # AUT Destination
#PIMA = pd.read_csv(r'C:\Users\user\Desktop\AUT Datasets\Creditcard.csv')
PIMA = pd.read_csv(r'C:\Users\Bobby\Desktop\AUT Datasets\PIMA_Indian_Diabetes.csv')  
print(PIMA)

# Explore the Data
PIMA.head(10)

# Explore the Data

PIMA.info()                          # [768 rows x 9 columns]

# Describe the Data

PIMA.describe()

#Code to check for shape of data
 
print("PIMA Diabetes data set dimensions : {}".format(PIMA.shape))     # 768 (Rows) and 9 (Columns)

#Code to check for any missing values

PIMA.isnull().any()
PIMA.isnull().values.any()
PIMA.isnull().sum()  
PIMA.isna().sum()

# Print Class values count of Target variable (Highly Imbalanced data)
# Class 0 means (Non-Diabetic) and Class 1 (Diabetic) 
# and ---> 500 (Non-Diabetic) and 268 (Diabetic)
                      
PIMA_count = PIMA['Class'].value_counts() 
print(PIMA_count)                   


# (Alternative) way to show Class variable Distribution

fig = sns.barplot(x = [1, 0], y = PIMA_count, data = PIMA, color = 'blue')
plt.ylabel('Frequency of Class variable')
plt.xlabel ('Class variable')
plt.show(fig)


#################################################### Creating Train/Test Split Data ##############################################
# Use Train/Test split with Random State Values and Stratified across both the classes (Diabetic and Non-Diabetic)
# Creating Input Features (X) and Target variable (y)

# Splitting the Dataframe into features and target variable 
y = PIMA['Class']
X = PIMA.drop(['Class'], axis = 1)          # Axis = 1 refers to Column

print (X)                                   # [768 rows x 8 columns]
print (y)                                   # 768


# Use Train/Test split with Random State Values
# Splitting the Data Set into 80% (Training) and 20% (Testing)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 123, stratify=y)

print(X_train.shape, X_test.shape)      # (614, 8) (154, 8)
print(y_train.shape, y_test.shape)      # (614,) (154,)

# Diabetic Vs Non-Diabetic

print('Labels counts in y:', np.bincount(y))                        #  [500 268]
print('Labels counts in y_train:', np.bincount(y_train))            #  [400 214]
print('Labels counts in y_test:', np.bincount(y_test))              #  [100  54]


######################################################## Separating Minority and Majority Classes ######################################################################
#  Separating Majority and Minority Classes in Training Set before Balancing
## Calculating the length of Minority and Majority Class in the entire PIMA Dataframe

# Print Class values count of Target variable (Highly Imbalanced data)
# Class 0 means (Non-Diabetic) and Class 1 (Diabetic) ---> 500 (Non-Diabetic) and 268 (Diabetic)

PIMA_Diabetic = len(PIMA[PIMA['Class']==1])                     # Length of Minority (Diabetic) Class 1
PIMA_NonDiabetic = len(PIMA[PIMA['Class']==0])                  # Length of Majority (Non-Diabetic) Class 0

print(PIMA_Diabetic)                             # 268
print(PIMA_NonDiabetic)                          # 500

#  Indices of the Majority (Non-Diabetic) Class

PIMA_NonDiabetic_index = PIMA[PIMA['Class']==0].index
print(len(PIMA_NonDiabetic_index))                       # 500   

# Indices of the Minority (Diabetic) Class

PIMA_Diabetic_index = PIMA[PIMA['Class']==1].index
print(len(PIMA_Diabetic_index))                     # 268    

##################################### Balancing Data in the Training Set ######################################################################

## Calculating the length of Minority and Majority Class in the Training Set

# The number of Majority Class in X_train 

PIMA_Xtrain_NDiabetic_index=[i for i in X_train.index if i in PIMA_NonDiabetic_index]
len(PIMA_Xtrain_NDiabetic_index)       #   400 (i in PIMA_NonDiabetic_index)
print(PIMA_Xtrain_NDiabetic_index)

# The number of Minority Class in X_train 

PIMA_Xtrain_Diabetic_index=[i for i in X_train.index if i not in PIMA_NonDiabetic_index]
len(PIMA_Xtrain_Diabetic_index)       # 214 (i in CC_Fraud_index)

# Downsampling Majority Class 

PIMA_Downsample = resample(PIMA_Xtrain_NDiabetic_index, replace = False, n_samples = 214, random_state = 123)
len(PIMA_Downsample)

# Randomly Sample the Majority Class wrt to Minority Class

Rand_index = np.random.choice(PIMA_Xtrain_NDiabetic_index, PIMA_Diabetic,replace='False')
len(Rand_index)

# Concatenate the Minority Indexes and Majority Indexes
PIMA_Undersample_index =  np.concatenate([PIMA_Xtrain_Diabetic_index,PIMA_Downsample])
len(PIMA_Undersample_index)  


######################################################## Undersample Dataset ######################################################################

# Balanced Undersampled Data

PIMA_Undersample = PIMA.iloc[PIMA_Undersample_index]
print (PIMA_Undersample)     

# PIMA Undersample Count

PIMA_Undersample_count = pd.value_counts(PIMA_Undersample['Class'])
print(PIMA_Undersample_count)
  
# PIMA Undersample Bar Chart

fig = PIMA_Undersample_count.plot(kind='bar')
plt.ylabel('Frequency of UnderSampled Class variables')
plt.xlabel ('Class Variables')
plt.show(fig)


# Use Train/Test split with Random State Values across both the classes (Diabetic and Non-Diabetic)

x_under = PIMA_Undersample.loc[:, PIMA_Undersample.columns!='Class']
y_under = PIMA_Undersample.loc[:, PIMA_Undersample.columns=='Class']
 
x_under_train, x_under_test, y_under_train, y_under_test = train_test_split(x_under, y_under, test_size=0.30, random_state=123)

print(x_under_train.shape, x_under_test.shape)      # (299, 8) (129, 8)
print(y_under_train.shape, y_under_test.shape)      # (299, 1) (129, 1)


######################################################## Logistic Regression ######################################################################
# with Undersample dataset
lr_under = LogisticRegression()
lr_under.fit(x_under_train, y_under_train)

lr_under_predict = lr_under.predict(x_under_test)
lr_under_accuracy = accuracy_score(lr_under_predict, y_under_test)
lr_recall = recall_score(lr_under_predict, y_under_test)

print(lr_under_accuracy)                                        
print(lr_recall)                                                

print(classification_report(y_under_test, lr_under_predict))
print(confusion_matrix(y_under_test, lr_under_predict))


######################################################## Support Vector Machine (SVM) ######################################################################

# Building the initial Model using Gaussian Kernel

SVM = SVC(C=1, kernel = 'rbf', random_state = 0)
SVM.fit(x_under_train, y_under_train)
SVM_Pred = SVM.predict(x_under_test)

SVM_Accuracy = accuracy_score(SVM_Pred, y_under_test)
print(SVM_Accuracy)                                       

# Recall of PIMA call Training set

SVM_recall =  recall_score(SVM_Pred, y_under_test)
print(SVM_recall)                                         


#  Alternatively 

# From Machine Learning Mastery (Expected, Predicted)

print(classification_report(y_under_test, SVM_Pred))
print(confusion_matrix(y_under_test, SVM_Pred))


################################################# Random Forest ######################################################################

RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 100)   # Change n_estimators to 100
RF.fit(x_under_train, y_under_train)
RF_Pred = RF.predict(x_under_test)


# Accuracy
RF_accuracy = accuracy_score(y_under_test, RF_Pred)
print (RF_accuracy)      # 0.7054263565891473


# Recall of PIMA 

RF_recall =  recall_score(RF_Pred, y_under_test)
print(RF_recall)        # 0.7076923076923077

# Confusion Matrix 

RF_cm = confusion_matrix(y_under_test, RF_Pred)
print(RF_cm)
pd.DataFrame(RF_cm, columns = ['Predicted Non-Fraud', 'Predicted Fraud'], index = ['True Non-Fraud', 'True Fraud'])

# Accuracy 

print("The Accuracy is " + str((RF_cm[0,0] + RF_cm[1,1])/(RF_cm[0,0] + RF_cm[0,1] + RF_cm[1,0]+RF_cm[1,1])* 100) + "%")     #  90.3553299492%%

 #Recall 
# The Recall is the Ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

print("The Recall from the confusion matrix is "+ str(RF_cm[1,1]/(RF_cm[1,0] + RF_cm[1,1])*100) +"%")   #87.74193548387098%%
    
# Precision 
# The Precision is the Ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The Precision is intuitively the ability of the classifier to find all the positive samples.

print("The Precision from the confusion matrix is "+ str(RF_cm[1,1]/(RF_cm[0,1] + RF_cm[1,1])*100) +"%")    # 94.8453608247%


# Print Classfication Report
print(classification_report(y_under_test,RF_Pred))

# Print Confusion Matrix
print(confusion_matrix(y_under_test,RF_Pred))


########################################  Decision Tree Classifier ######################################################################

#  For Undersampled Data  

# Decision Tree Classifier

DT = DecisionTreeClassifier()
DT.fit(x_under_train, y_under_train)
DT_Pred = DT.predict(x_under_test)

# Accuracy

DT_accuracy = accuracy_score(y_under_test, DT_Pred)
print (DT_accuracy)             # 0.6046511627906976


# Recall of PIMA 

DT_recall =  recall_score(DT_Pred, y_under_test)
print(DT_recall)        # 0.7076923076923077


                          #   0.9155405405405406

# Confusion Matrix 

DT_cm = confusion_matrix(y_under_test, DT_Pred)
print(DT_cm)
pd.DataFrame(DT_cm, columns = ['Predicted Non-Fraud', 'Predicted Fraud'], index = ['True Non-Fraud', 'True Fraud'])

# Accuracy 
  
print("The Accuracy is " + str((RF_cm[0,0] + DT_cm[1,1])/(DT_cm[0,0] + DT_cm[0,1] + DT_cm[1,0]+DT_cm[1,1])* 100) + "%")     #  90.3553299492%%

 #Recall 
# The Recall is the Ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

print("The Recall from the confusion matrix is "+ str(DT_cm[1,1]/(DT_cm[1,0] + DT_cm[1,1])*100) +"%")   # 91.61290322580645%
    
# Precision 
# The Precision is the Ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The Precision is intuitively the ability of the classifier to find all the positive samples.

print("The Precision from the confusion matrix is "+ str(DT_cm[1,1]/(DT_cm[0,1] + DT_cm[1,1])*100) +"%")    # 94.8453608247%

# Print Classfication Report

print(classification_report(y_under_test,DT_Pred))

# Print Confusion Matrix
print(confusion_matrix(y_under_test,DT_Pred))


##########################  Predict on Full Dataset using Undersampled Dataset #############################################


lr_pred_full = lr_under.predict(X_test)

# print(recall_score(y_test,y_pred_full, average = None)) 
print(accuracy_score(y_test,lr_pred_full))                    # 0.7857142857142857
print(recall_score(y_test,lr_pred_full))                      # 0.7592592592592593


print(classification_report(y_test,lr_pred_full))
print(confusion_matrix(y_test,lr_pred_full))
 
##### SVM

SVM_pred_full = SVM.predict(X_test)

print(recall_score(y_test,SVM_pred_full))                      # 0.037037037037037035
print(accuracy_score(y_test,SVM_pred_full))                    # 0.6623376623376623

print(classification_report(y_test,SVM_pred_full))
print(confusion_matrix(y_test,SVM_pred_full))

##### RF

RF_predicted_full = RF.predict(X_test)
print(recall_score(y_test,RF_predicted_full))                 # 0.6296296296296297
print(accuracy_score(y_test,RF_predicted_full))               # 0.7337662337662337

print(classification_report(y_test,RF_predicted_full))
print(confusion_matrix(y_test,RF_predicted_full))

##### Decision Tree

DT_predicted_full =  DT.predict(X_test)
print(recall_score(y_test,DT_predicted_full))                  # 0.6481481481481481
print(accuracy_score(y_test,DT_predicted_full))                # 0.6558441558441559

print(classification_report(y_test,DT_predicted_full))
print(confusion_matrix(y_test,DT_predicted_full))

############################################## Upsample Dataset #######################################################################


# Upsampling Minority Class

PIMA_Upsample = resample(PIMA_Xtrain_Diabetic_index, replace = True, n_samples = 400, random_state = 123)
len(PIMA_Upsample)

# Concatenate the Minority Indexes and Majority Indexes
PIMA_Upsample_index =  np.concatenate([PIMA_Xtrain_NDiabetic_index,PIMA_Upsample])
len(PIMA_Upsample_index)  


# Balanced Upsampled Data
PIMA_Upsample = PIMA.iloc[PIMA_Upsample_index]
print (PIMA_Upsample)                                                      # [454902 rows x 30 columns]]

# Credit Card Oversample Count

PIMA_Upsample_count = pd.value_counts(PIMA_Upsample['Class'])
print(PIMA_Upsample_count)
  
# CreditCard Undersample Bar Chart

fig = PIMA_Upsample_count.plot(kind='bar')
plt.ylabel('Frequency of UnderSampled Class variables')
plt.xlabel ('Class Variables')
plt.show(fig)

# Use Train/Test split with Random State Values across both the classes (Fraud and Non-Fraud)

x_up = PIMA_Upsample.loc[:, PIMA_Upsample.columns!='Class']
y_up = PIMA_Upsample.loc[:, PIMA_Upsample.columns=='Class']
 
x_up_train, x_up_test, y_up_train, y_up_test = train_test_split(x_up, y_up, test_size=0.20, random_state=123)

print(x_up_train.shape, x_up_test.shape)      # (640, 8) (160, 8)
print(y_up_train.shape, y_up_test.shape)      # (640, 1) (160, 1)



######################################################## Logistic Regression ######################################################################

lr_up = LogisticRegression()
lr_up.fit(x_up_train, y_up_train)

lr_up_predict = lr_up.predict(x_up_test)
lr_up_accuracy = accuracy_score(lr_up_predict, y_up_test)
lr_up_recall = recall_score(lr_up_predict, y_up_test)

print(lr_up_accuracy)                                              # 0.71875
print(lr_up_recall)                                                # 0.6883116883116883
print(classification_report(y_up_test, lr_up_predict))
print(confusion_matrix(y_up_test, lr_up_predict))

######################################################## Support Vector Machine (SVM) ######################################################################

# Building the initial Model using Gaussian Kernel

SVM_up = SVC(C=1, kernel = 'rbf', random_state = 0)
SVM_up.fit(x_up_train, y_up_train)

SVM_Pred_up = SVM_up.predict(x_up_test)


# Accuracy of Training (Label) test data (y_test) 

SVM_Accuracy_up = accuracy_score(y_up_test, SVM_Pred_up)
print(SVM_Accuracy_up)                                        # 0.925

# Recall of Credit call Training set

SVM_classifier_recall_up =  recall_score(y_up_test, SVM_Pred_up)
print(SVM_classifier_recall_up)                      # 0.8378378378378378 
#  Alternatively 

# From Machine Learning Mastery (Expected, Predicted)

print('Accuracy Score: ', accuracy_score(y_up_test, SVM_Pred_up))        #     # 0.925
print('Recall Score: ', recall_score(y_up_test, SVM_Pred_up))            #  0.8903225806451613

print(classification_report(y_up_test, SVM_Pred_up))
print(confusion_matrix(y_up_test, SVM_Pred_up))


################################################# Random Forest ######################################################################


RF_up = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 100)   # Change n_estimators to 100

# RF with downsampled Data

RF_up.fit(x_up_train, y_up_train)
RF_Pred_up = RF_up.predict(x_up_test)
RF_accuracy = accuracy_score(y_up_test, RF_Pred_up)
print (RF_accuracy)      # 0.8625


# Print Classfication Report
print(classification_report(y_up_test,RF_Pred_up))


# Print Confusion Matrix
print(confusion_matrix(y_up_test,RF_Pred_up))


########################################  Decision Tree Classifier ######################################################################

# RF = RandomForestClassifier(n_estimators = 30, max_depth = 10oob_score = True, random_state = 100)
# n_estimator:  The number of trees in the random forest classification
# Criterion: Loss fucntion used to measure the quality of split 
# Random State:  See used by the Random State Generator for randomising dataset

#  Splitting the Dataset into Training (80%) and Testing (20%)


# Decision Tree Classifier

DT_up = DecisionTreeClassifier()

# DT 
DT_up.fit(x_up_train, y_up_train)

# Evaluation on test data
DT_Pred_up = DT.predict(x_up_test)


# Accuracy of Training (Label) test data (y_test) 

DT_accuracy = accuracy_score(y_up_test, DT_Pred_up)
print (DT_accuracy)                               #   0.9125

## Confusion Matrix 
#
#DT_cm = confusion_matrix(y, DT_predicted)
#print(DT_cm)
#pd.DataFrame(DT_cm, columns = ['Predicted Non-Fraud', 'Predicted Fraud'], index = ['True Non-Fraud', 'True Fraud'])
#
## Accuracy 
#  
#print("The Accuracy is " + str((RF_cm[0,0] + DT_cm[1,1])/(DT_cm[0,0] + DT_cm[0,1] + DT_cm[1,0]+DT_cm[1,1])* 100) + "%")     #  90.3553299492%%
#
# #Recall 
## The Recall is the Ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
#
#print("The Recall from the confusion matrix is "+ str(DT_cm[1,1]/(DT_cm[1,0] + DT_cm[1,1])*100) +"%")   #86.7924528302%
#    
#recall_score(y_test, DT_predicted, average = 'macro')      # 0.8773575764014103
#recall_score(y_test, DT_predicted, average = 'micro')      # 0.9991924440855307
#recall_score(y_test, DT_predicted, average = 'weighted')   # 0.9991924440855307

# Precision 
# The Precision is the Ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The Precision is intuitively the ability of the classifier to find all the positive samples.

print("The Precision from the confusion matrix is "+ str(DT_cm[1,1]/(DT_cm[0,1] + DT_cm[1,1])*100) +"%")    # 94.8453608247%

# Print Classfication Report

print(classification_report(y_up_test,DT_Pred_up))


# Print Confusion Matrix
print(confusion_matrix(y_up_test,DT_Pred_up))


##########################  Predict on Full Dataset using Upsampled Dataset #############################################


lr_full_pred = lr_up.predict(X_test)

# print(recall_score(y_test,y_pred_full, average = None)) 
print(recall_score(y_test,lr_full_pred))                      # 0.7592592592592593
print(accuracy_score(y_test,lr_full_pred))                    # 0.7922077922077922

print(classification_report(y_test,lr_full_pred))
print(confusion_matrix(y_test,lr_full_pred))

##### SVM

SVM_full_pred = SVM_up.predict(X_test)

print(recall_score(y_test,SVM_pred_full))                      # 0.037037037037037035
print(accuracy_score(y_test,SVM_pred_full))                    # 0.6623376623376623

print(classification_report(y_test,SVM_pred_full))
print(confusion_matrix(y_test,SVM_full_pred))
 
##### RF

RF_full_pred = RF.predict(X_test)
print(recall_score(y_test,RF_full_pred))                 # 0.6296296296296297
print(accuracy_score(y_test,RF_full_pred))               # 0.7337662337662337

print(classification_report(y_test,RF_full_pred))
print(confusion_matrix(y_test,RF_full_pred))

### Decision Tree

DT_full_pred =  DT.predict(X_test)
print(recall_score(y_test,DT_full_pred))                  # 0.6481481481481481
print(accuracy_score(y_test,DT_full_pred))                # 0.6558441558441559

print(classification_report(y_test,DT_full_pred))
print(confusion_matrix(y_test,DT_full_pred))

