
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Analysis of Bank Marketign dataset using Standard ML Classification Algorithms (Stratification)
 
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

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

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


# Creating a Bank Dataframe

Bank = pd.read_csv(r'C:\Users\Bobby\Desktop\AUT Datasets\BankFull.csv', delimiter = ";")  

# Explore the Data

Bank.info()                          # 41188 (Rows) and 21 (Columns)  (10 Numerical Variables and 11 Categorical variables)


# Describe the Data  
# The Columns 'balance', 'duration', 'campaign', 'pdays', 'previous' have outliers. 


Bank.describe()                    # Only gives the Numerical variables


#Code to check for shape of data
 
print ("Number of rows:  ", Bank.shape)         # (41188, 21) 
print ("Number of rows:  ", Bank.shape[0])      # 41188 
print ("Number of Columns:  ", Bank.shape[1])   # 21 

# Rename the Target Variable to Deposit from 'y' to make it easier 

Bank = Bank.rename(columns = {'y': 'Deposit'})

# Head 

Bank.head(10)


#Code to check for any missing values

Bank.isnull().any()
Bank.isnull().values.any()
Bank.isnull().sum()  


# Print Class values count of Target variable (Highly Imbalanced data)
# Replacing 'Yes' ---> Class 1 and No ----> Class 0 
# There were 36548 (89%) instances of No (Class 0) and 4640 instances of Yes (11%) (Class 1) 

Bank.Deposit.replace(['no', 'yes'], [0, 1], inplace=True)     
Bank_count = Bank['Deposit'].value_counts() 
print(Bank_count)      
    

# Plot of Class variable distribution

fig = sns.barplot(x = [0,1], y = Bank_count, data = Bank, color = 'blue')
plt.ylabel('Frequency of Class variable')
plt.xlabel ('Class variable')
plt.show(fig)


# Education 
Bank['education'].unique()



# Group basic.4y", "basic.9y" and "basic.6y" as "basic".
Bank['education']=np.where(Bank['education'] =='basic.9y', 'Basic', Bank['education'])
Bank['education']=np.where(Bank['education'] =='basic.6y', 'Basic', Bank['education'])
Bank['education']=np.where(Bank['education'] =='basic.4y', 'Basic', Bank['education'])


# Groupby by Deposit
Bank.groupby('Deposit').mean()

# Groupby by Job
Bank.groupby('job').mean()


# Some Exploratory Data visualisation 

# Outcome based on Jobs

pd.crosstab(Bank.job,Bank.Deposit).plot(kind='bar')
plt.title('Job Title')
plt.xlabel('Job')
plt.ylabel('Term Deposit Purchase Count')


# Outcome based on Education

pd.crosstab(Bank.education,Bank.Deposit).plot(kind='bar')
plt.title('Education Level')
plt.xlabel('Education')
plt.ylabel('Term Deposit Purchase Count')


# Outcome based on Age

Bank.age.hist()
plt.title('Age Group')
plt.xlabel('Age')
plt.ylabel('Frequency')


# Creating Dummy variables for Categorical variables

Categ_var =['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in Categ_var:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(Bank[var], prefix=var)
    data1=Bank.join(cat_list)
    Bank=data1
    
Categ_var=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=Bank.columns.values.tolist()
Good_data=[i for i in data_vars if i not in Categ_var]
Bank_final=Bank[Good_data]
Bank_final.columns.values


#################################################### Creating Train/Test Split Data ######################################

# Use Train/Test split with Random State Values and Stratified across both the classes (Fraud and Non-Fraud)
# Creating Input Features (X) and Target variable (y)

# Splitting into Feature and Target Variable

X = Bank_final.loc[:, Bank_final.columns != 'Deposit']
y = Bank_final.loc[:, Bank_final.columns == 'Deposit']


# Use Train/Test split with Random State Values
# Splitting the Data Set into 80% (Training) and 20% (Testing)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 123, stratify=y)

print(X_train.shape, X_test.shape)     # (32950, 61) (8238, 61)
print(y_train.shape, y_test.shape)     # (32950, 1) (8238, 1)

print(X_train)
print(len(X_train))


######################################################## Separating Minority and Majority Classes ######################################################################
#  Separating Majority and Minority Classes in Training Set before Balancing

## Calculating the length of Minority and Majority Class in the entire CreditCard Dataframe

# Print Class values count of Target variable (Highly Imbalanced data)
# Replacing 'Yes' ---> Class 1 and No ----> Class 0 
# There were 36548 (89%) instances of No (Class 0) and 4640 instances of Yes (11%) (Class 1) 

Bank_No = len(Bank_final[Bank_final['Deposit']==0])                      # Length of Majority Class (No)
Bank_Yes = len(Bank_final[Bank_final['Deposit']==1])                     # Length of Minority Class (Yes)

print(Bank_No)                             # 36548
print(Bank_Yes)                            # 4640


#  Indices of the Majority (No) Class

Bank_No_index = Bank_final[Bank_final['Deposit']==0].index
print(len(Bank_No_index))                 # 36548   

# Indices of the Minority (Yes) Class

Bank_Yes_index = Bank_final[Bank_final['Deposit']==1].index
print(len(Bank_Yes_index))               # 4640

##################################### Balancing Data in the Training Set ######################################################################

## Calculating the length of Minority and Majority Class in the Training Set
## Undersample
# The number of Majority Class in X_train 

Bank_Xtrain_No_index=[i for i in X_train.index if i in Bank_No_index]
len(Bank_Xtrain_No_index)       #  29238 (i in Bank_No_index)

# The number of Minority Class in X_train 

Bank_Xtrain_Yes_index=[i for i in X_train.index if i not in Bank_No_index]
len(Bank_Xtrain_Yes_index)       # 3712 

# Downsampling Majority Class 

Bank_Downsample = resample(Bank_Xtrain_No_index, replace = False, n_samples = 3712, random_state = 123)
len(Bank_Downsample)            # 3712

# Concatenate the Minority Indexes and Majority Indexes
Bank_Undersample_index =  np.concatenate([Bank_Xtrain_Yes_index,Bank_Downsample])
len(Bank_Undersample_index)                    # 7424



######################################################## Undersample Dataset ######################################################################

# Balanced Undersampled Data

Bank_Undersample = Bank_final.iloc[Bank_Undersample_index]
print (Bank_Undersample)                             # [7424 rows x 62 columns]

# Credit Card Undersample Count

Bank_Undersample_count = pd.value_counts(Bank_Undersample['Deposit'])
print(Bank_Undersample_count)
  
# CreditCard Undersample Bar Chart

fig = Bank_Undersample_count.plot(kind='bar')
plt.ylabel('Frequency of UnderSampled Class variables')
plt.xlabel ('Class Variables')
plt.show(fig)


# Use Train/Test split with Random State Values across both the classes (Fraud and Non-Fraud)

x_under = Bank_Undersample.loc[:, Bank_Undersample.columns!='Deposit']
y_under = Bank_Undersample.loc[:, Bank_Undersample.columns=='Deposit']
 
x_under_train, x_under_test, y_under_train, y_under_test = train_test_split(x_under, y_under, test_size=0.20, random_state=123)

print(x_under_train.shape, x_under_test.shape)      # (5939, 61) (1485, 61)
print(y_under_train.shape, y_under_test.shape)      # (5939, 1) (1485, 1)


######################################################## Logistic Regression ######################################################################

# with Undersample dataset
lr_under = LogisticRegression()
lr_under.fit(x_under_train, y_under_train.values.ravel())

lr_under_predict = lr_under.predict(x_under_test)
lr_under_accuracy = accuracy_score(lr_under_predict, y_under_test)
lr_recall = recall_score(lr_under_predict, y_under_test)

print(lr_under_accuracy)                                        # 0.9620253164556962
print(lr_recall)                                                # 0.984

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

print(accuracy_score(y_under_test,RF_Pred))                # 0.9001088444928198
print(recall_score(y_under_test,RF_Pred))                  # 0.9527027027027027
print(precision_score(y_under_test,RF_Pred))               # 0.015088449531737774

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

print(accuracy_score(y_under_test,DT_Pred))                # 0.9001088444928198
print(recall_score(y_under_test,DT_Pred))                  # 0.9527027027027027
print(precision_score(y_under_test,DT_Pred))               # 0.015088449531737774

# Print Classification and Confusion Matrix 
print(classification_report(y_under_test,DT_Pred))
print(confusion_matrix(y_under_test,DT_Pred))


##########################  Predict on Full Dataset using Undersampled Dataset #############################################


lr_pred_full = lr_under.predict(X_test)

# print(recall_score(y_test,y_pred_full, average = None)) 

print(accuracy_score(y_test,lr_pred_full))                    # 0.8602816217528526
print(recall_score(y_test,lr_pred_full))                      # 0.8775510204081632
print(precision_score(y_test,lr_pred_full))                   # 0.43956639566395667

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

print(accuracy_score(y_test,RF_predicted_full))               # 0.8622238407380433
print(recall_score(y_test,RF_predicted_full))                 # 0.8760775862068966
print(precision_score(y_test,RF_predicted_full))              # 0.44353518821603927

# Print Classification and Confusion Matrix 
print(classification_report(y_test,RF_predicted_full))
print(confusion_matrix(y_test,RF_predicted_full))

##### Decision Tree

DT_predicted_full =  DT.predict(X_test)

print(accuracy_score(y_test,DT_predicted_full))                # 0.8431658169458607
print(recall_score(y_test,DT_predicted_full))                  # 0.834051724137931
print(precision_score(y_test,DT_predicted_full))               # 0.40481171548117156

# Print Classification and Confusion Matrix 
print(classification_report(y_test,DT_predicted_full))
print(confusion_matrix(y_test,DT_predicted_full))


############################################## Upsample Dataset #######################################################################


# Upsampling Minority Class

Bank_Upsample = resample(Bank_Xtrain_Yes_index, replace = True, n_samples = 29238, random_state = 123)
len(Bank_Upsample)

# Concatenate the Minority Indexes and Majority Indexes
Bank_Upsample_index =  np.concatenate([Bank_Xtrain_No_index,Bank_Upsample])
len(Bank_Upsample_index)               # 58476


# Balanced Upsampled Data
Bank_Upsample = Bank_final.iloc[Bank_Upsample_index]
print (Bank_Upsample)                                                      # [58476 rows x 62 columns]]

# Credit Card Oversample Count

Bank_Upsample_count = pd.value_counts(Bank_Upsample['Deposit'])
print(Bank_Upsample_count)
  
# CreditCard Undersample Bar Chart

fig = Bank_Upsample_count.plot(kind='bar')
plt.ylabel('Frequency of Upsampled Class variables')
plt.xlabel ('Class Variables')
plt.show(fig)


# Use Train/Test split with Random State Values across both the classes (Fraud and Non-Fraud)

x_up = Bank_Upsample.loc[:, Bank_Upsample.columns!='Deposit']
y_up = Bank_Upsample.loc[:, Bank_Upsample.columns=='Deposit']
 
x_up_train, x_up_test, y_up_train, y_up_test = train_test_split(x_up, y_up, test_size=0.20, random_state=123)

print(x_up_train.shape, x_up_test.shape)      # (46780, 61) (11696, 61)
print(y_up_train.shape, y_up_test.shape)      # (46780, 1) (11696, 1)



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
print(recall_score(y_test,lr_full_pred))                 # 0.8803879310344828
print(precision_score(y_test,lr_full_pred))              # 0.4401939655172414

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

print(accuracy_score(y_test,RF_full_pred))               # 0.8622238407380433
print(recall_score(y_test,RF_full_pred))                 # 0.8760775862068966
print(precision_score(y_test,RF_full_pred))              # 0.44353518821603927

print(classification_report(y_test,RF_full_pred))
print(confusion_matrix(y_test,RF_full_pred))

### Decision Tree

DT_full_pred =  DT.predict(X_test)

print(accuracy_score(y_test,DT_full_pred))                # 0.8431658169458607
print(recall_score(y_test,DT_full_pred))                  # 0.834051724137931
print(precision_score(y_test,DT_full_pred))               # 0.40481171548117156

print(classification_report(y_test,DT_full_pred))
print(confusion_matrix(y_test,DT_full_pred))





