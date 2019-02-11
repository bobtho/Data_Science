# -*- coding: utf-8 -*-
"""
Analysis of Parkisons using Standard ML Classification Algorithms below
 
      1) Logistic Regression 
      2) SVM
      3) Random Forest
      4) Decision Tree 
    
""" 


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

# Creating a Parkinsons Dataframe

#CreditCard = pd.read_csv(r'H:\AUT Datasets\Creditcard.csv')                              # AUT Destination
#CreditCard = pd.read_csv(r'C:\Users\user\Desktop\AUT Datasets\Creditcard.csv')
Parkinson = pd.read_csv(r'C:\Users\Bobby\Desktop\AUT Datasets\LSVT_Voice.csv')  
print(Parkinson)



# Explore the Data

Parkinson.info()                          # [126 rows x 314 columns]

# Describe the Data

Parkinson.describe()

#Code to check for shape of data
 
print ("Number of rows:  ", Parkinson.shape[0])   # Gives the number of (Rows) only 126 Instances 
print ("Number of rows:  ", Parkinson.shape)       # 126 (Rows) and 314 (Columns)

#Code to check for any missing values

Parkinson.isnull().any()
Parkinson.isnull().values.any()
Parkinson.isnull().sum() 
Parkinson.isna().sum() 

# Print Class values count of Target variable (Highly Imbalanced data)
# Original Dataset has 42 instances of Class 1 (“Acceptable") and 84 instances of Class 2 (“Unacceptable”) in terms of their performance. 
# Modified Class 1 --> 0 and Class 2 --> 1.  In this new scenario, there are 42 instances of Acceptable (Class 0) and 84 instances of Unacceptable (Class 1) 

Parkinson = Parkinson.drop(['Subject_index', 'Age', 'Gender'], axis=1) 
Parkinson.Class.replace([1, 2], [0, 1], inplace=True)                            # Replace Class 1 --> 0 and Class 2 ---> 1     
Parkinson_count = Parkinson['Class'].value_counts() 
print(Parkinson_count)

print(Parkinson)                                                                 # [126 rows x 311 columns]



# (Alternative) way to show Class variable Distribution

fig = sns.barplot(x = [1, 0], y = Parkinson_count, data = Parkinson, color = 'blue')
plt.ylabel('Frequency of Class variable')
plt.xlabel ('Class variable')
plt.ylabel('Frequency of Class variable')
plt.xlabel ('Class variable')
plt.show(fig)

yn = Parkinson['Class']
Xn = Parkinson.drop(['Class'], axis = 1)    # Axis = 1 refers to Column

print (Xn)            # [126 rows x 310 columns]
print (yn)            # 126
  
# Use Train/Test split with Random State Values
# Splitting the Data Set into 80% (Training) and 20% (Testing)

X_train, X_test, y_train, y_test = train_test_split(Xn, yn, test_size = 0.20, random_state = 123, stratify=yn)

print(X_train.shape, X_test.shape)      # (100, 310) (26, 310)
print(y_train.shape, y_test.shape)      # (100,) (26,)

# Print counts
# Non-Fradulent Vs Fradulent

print('Labels counts in y:', np.bincount(yn))                        #  [42 84] 
print('Labels counts in y_train:', np.bincount(y_train))            #  [33 67]
print('Labels counts in y_test:', np.bincount(y_test))              #  [ 9 17]


#################################################### with RFECE ################

from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
log4 = LogisticRegression() 
rfecv = RFECV(estimator=log4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(X_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])


# Plot number of features VS. cross-validation scores
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


###################   Using RFE   ############################

from sklearn.feature_selection import RFE
# Create the RFE object and rank each feature
log3 = LogisticRegression()     
rfe = RFE(estimator=log3, n_features_to_select=8, step=1)
rfe = rfe.fit(X_train, y_train)

print(rfe.support_)
print(rfe.ranking_)

Lr_pred = rfe.predict(X_test) 

# Feature Selection using RFE

Best_features = list(zip(Xn, rfe.support_))

new_features = []
for key, value in enumerate(Best_features):
    if(value[1]) == True:
        new_features.append(value[0])

        
print(new_features)


# Creating a dataframe with new features


Parkinson_new = pd.concat([Parkinson[new_features], yn],axis=1)

print(Parkinson_new)    

# Splitting Features and 

y = Parkinson_new['Class']
X = Parkinson_new.drop(['Class'], axis = 1)  

print(X)          #  [126x 8]
print(y)   
#   Creating Train/Test Dataset

Xn_train, Xn_test, yn_train, yn_test = train_test_split(X, y, test_size = 0.20, random_state = 123, stratify=y)

print(Xn_train.shape, Xn_test.shape)      # (100, 8) (26, 8)  (100, 9) (26, 9)
print(yn_train.shape, yn_test.shape)      # (100,) (26,)  (100,) (26,)


######################################################## Separating Minority and Majority Classes ######################################################################
#  Separating Majority and Minority Classes in Training Set before Balancing
# Print Class values count of Target variable (Highly Imbalanced data)
# Original Dataset has 42 instances of Class 1 (“Acceptable") and 84 instances of Class 2 (“Unacceptable”) in terms of their performance. 
# Modified Class 1 --> 0 and Class 2 --> 1.  In this new scenario, there are 42 instances of Acceptable (Class 0) and 84 instances of Unacceptable (Class 1) 


Parkinson_Acc = len(Parkinson[Parkinson['Class']==0])                    # Length of Minority (Acceptable) Class 0
Parkison_Unacc = len(Parkinson[Parkinson['Class']==1])                  # Length of Majority (Unaccepable) Class 1

print(Parkinson_Acc)                           # 42
print(Parkison_Unacc)                          # 84

#  Indices of the  Minority (Acceptable) Class Class

Parkison_Acc_index = Parkinson[Parkinson['Class']==0].index
print(len(Parkison_Acc_index))                       # 42   

# Indices of the Majority (Unacceptable) Class

Parkison_Unacc_index = Parkinson[Parkinson['Class']==1].index
print(len(Parkison_Unacc_index))                     # 84    

##################################### Balancing Data in the Training Set ######################################################################

## Calculating the length of Minority and Majority Class in the Training Set

# The number of Majority Class in X_train 

Parkinson_Xntrain_Un_index=[i for i in Xn_train.index if i in Parkison_Unacc_index]
len(Parkinson_Xntrain_Un_index)       #   67 (i in PIMA_NonDiabetic_index)
print(Parkinson_Xntrain_Un_index)

# The number of Minority Class in X_train 

Parkinson_Xntrain_Acc_index=[i for i in Xn_train.index if i not in Parkison_Unacc_index]
len(Parkinson_Xntrain_Acc_index)       # 33 (i in CC_Fraud_index)

# Downsampling Majority Class 
Parkinson_Downsample = resample(Parkinson_Xntrain_Un_index, replace = False, n_samples = 33, random_state = 123)
len(Parkinson_Downsample)

# Randomly Sample the Majority Class wrt to Minority Class

Rand_index = np.random.choice(Parkinson_Xntrain_Un_index, Parkinson_Acc,replace='False')
len(Rand_index)

# Concatenate the Minority Indexes and Majority Indexes
Parkinson_Undersample_index =  np.concatenate([Parkinson_Xntrain_Acc_index,Parkinson_Downsample])
len(Parkinson_Undersample_index)  


######################################################## Undersample Dataset ######################################################################

# Balanced Undersampled Data

Parkinson_Undersample = Parkinson_new.iloc[Parkinson_Undersample_index]
print (Parkinson_Undersample)     

# PIMA Undersample Count

Parkinson_Undersample_count = pd.value_counts(Parkinson_Undersample['Class'])
print(Parkinson_Undersample_count)
  
# PIMA Undersample Bar Chart

fig = Parkinson_Undersample_count.plot(kind='bar')
plt.ylabel('Frequency of UnderSampled Class variables')
plt.xlabel ('Class Variables')
plt.show(fig)


# Use Train/Test split with Random State Values across both the classes (Diabetic and Non-Diabetic)

x_under = Parkinson_Undersample.loc[:, Parkinson_Undersample.columns!='Class']
y_under = Parkinson_Undersample.loc[:, Parkinson_Undersample.columns=='Class']
 
x_under_train, x_under_test, y_under_train, y_under_test = train_test_split(x_under, y_under, test_size=0.20, random_state=123)

print(x_under_train.shape, x_under_test.shape)      # (52, 8) (14, 8)
print(y_under_train.shape, y_under_test.shape)      # (52, 1) (14, 1)


######################################################## Logistic Regression ######################################################################
# with Undersample dataset
lr_under = LogisticRegression()
lr_under.fit(x_under_train, y_under_train)

lr_under_predict = lr_under.predict(x_under_test)
lr_under_accuracy = accuracy_score(lr_under_predict, y_under_test)
lr_recall = recall_score(lr_under_predict, y_under_test)

print(lr_under_accuracy)                                        # 0.7209302325581395
print(lr_recall)                                                # 0.7164179104477612

print(classification_report(y_under_test, lr_under_predict))
print(confusion_matrix(y_under_test, lr_under_predict))


######################################################## Support Vector Machine (SVM) ######################################################################

# Building the initial Model using Gaussian Kernel

SVM = SVC(C=1, kernel = 'linear', random_state = 0)
SVM.fit(x_under_train, y_under_train)
SVM_Pred = SVM.predict(x_under_test)

SVM_Accuracy = accuracy_score(SVM_Pred, y_under_test)
print(SVM_Accuracy)                                        # 0.5038759689922481

# Recall of PIMA call Training set

SVM_recall =  recall_score(SVM_Pred, y_under_test)
print(SVM_recall)                                          # 0.9606299212598425


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

# RF = RandomForestClassifier(n_estimators = 30, max_depth = 10oob_score = True, random_state = 100)
# n_estimator:  The number of trees in the random forest classification
# Criterion: Loss fucntion used to measure the quality of split 
# Random State:  See used by the Random State Generator for randomising dataset


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


lr_pred_full = lr_under.predict(Xn_test)

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

Parkinson_Upsample = resample(Parkinson_Xntrain_Acc_index, replace = True, n_samples = 67, random_state = 123)
len(Parkinson_Upsample)

# Concatenate the Minority Indexes and Majority Indexes
Parkinson_Upsample_index =  np.concatenate([Parkinson_Xntrain_Un_index,Parkinson_Upsample])
len(Parkinson_Upsample_index)  


# Balanced Upsampled Data
Parkinson_Upsample = Parkinson.iloc[Parkinson_Upsample_index]
print (Parkinson_Upsample)                                                      # [454902 rows x 30 columns]]

# Credit Card Oversample Count

Parkinson_Upsample_count = pd.value_counts(Parkinson_Upsample['Class'])
print(Parkinson_Upsample_count)
  
# CreditCard Undersample Bar Chart

fig = Parkinson_Upsample_count.plot(kind='bar')
plt.ylabel('Frequency of UnderSampled Class variables')
plt.xlabel ('Class Variables')
plt.show(fig)

# Use Train/Test split with Random State Values across both the classes (Fraud and Non-Fraud)

x_up = Parkinson_Upsample.loc[:, Parkinson_Upsample.columns!='Class']
y_up = Parkinson_Upsample.loc[:, Parkinson_Upsample.columns=='Class']
 
x_up_train, x_up_test, y_up_train, y_up_test = train_test_split(x_up, y_up, test_size=0.20, random_state=123)

print(x_up_train.shape, x_up_test.shape)      # (107, 310) (27, 310)
print(y_up_train.shape, y_up_test.shape)      # (107, 1) (27, 1)



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


RF_recall =  recall_score(y_up_test, RF_Pred_up)
print(RF_recall)  

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

# Recall Score
DT_recallscore = recall_score(y_up_test, DT_Pred_up)
print (DT_recallscore) 

# Confusion Matrix 

DT_cm = confusion_matrix(y, DT_predicted)
print(DT_cm)
pd.DataFrame(DT_cm, columns = ['Predicted Non-Fraud', 'Predicted Fraud'], index = ['True Non-Fraud', 'True Fraud'])

# Accuracy 
  
print("The Accuracy is " + str((RF_cm[0,0] + DT_cm[1,1])/(DT_cm[0,0] + DT_cm[0,1] + DT_cm[1,0]+DT_cm[1,1])* 100) + "%")     #  90.3553299492%%

 #Recall 
# The Recall is the Ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

print("The Recall from the confusion matrix is "+ str(DT_cm[1,1]/(DT_cm[1,0] + DT_cm[1,1])*100) +"%")   #86.7924528302%
    
recall_score(y_test, DT_predicted, average = 'macro')      # 0.8773575764014103
recall_score(y_test, DT_predicted, average = 'micro')      # 0.9991924440855307
recall_score(y_test, DT_predicted, average = 'weighted')   # 0.9991924440855307

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