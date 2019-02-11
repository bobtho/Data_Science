# -*- coding: utf-8 -*-
"""

Analysis of Census dataset using Keras (HyperTuned)

"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

import seaborn as sns
from sklearn.model_selection import train_test_split


from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
import keras as kr

from keras.models import Model,Sequential, load_model
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier

from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras.utils import to_categorical
from keras.regularizers import L1L2

# Creating a Census Dataframe

#Census = pd.read_csv(r'H:\AUT Datasets\Census.csv')                              # AUT Destination
#Census = pd.read_csv(r'C:\Users\user\Desktop\AUT Datasets\Census.csv')
Census = pd.read_csv(r'C:\Users\Bobby\Desktop\AUT Datasets\AdultIncome.csv') 
print(Census)

# Explore the Data

Census.info()                          # 284807 (Rows) and 31 (Columns)

# Describe the Data

Census.describe()
Census.columns

#Code to check for shape of data
 
print ("Number of rows:  ", Census.shape[0])   # Gives the number of (Rows) only  284807 Instances 
print ("Number of rows:  ", Census.shape[1])   # Number of Columns (31)
print ("Number of rows:  ", Census.shape)      #284807 Instances (Rows)  and 31 Features (Columns) 

#Code to check for any missing values

Census.isnull().any()
Census.isnull().values.any()
Census.isnull().sum()  

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


#Converting Categorical variables into Quantitative variables

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


# Fill missing Categorial Values

Census["workclass"] = Census["workclass"].fillna("X")
Census["native.country"] = Census["native.country"].fillna("United-States")


# Drop data

Census.drop(labels=["workclass","education","relationship","race","native.country"], axis = 1, inplace = True)
print(Census.head())

Census.head(100)



# Creating a new Column with Normalised Amount and removing the Amount and Time Column in DataFrame

#################################################### Creating Train/Test Split Data ######################################

# Create 80/20 (Train/Test) split with Random State Values across both the classes (Fraud and Non-Fraud)
# Creating Input Features (X) and Target variable (y)

X_train, X_test = train_test_split(Census, test_size = 0.20, random_state = 123)

# Create Training Set on < 50K Class (Class 0)

X_train = X_train[X_train.income == 0]
y_train = X_train['income']
X_train = X_train.drop(['income'], axis=1)

# Create Test set on < 50K Class (Class 0)

y_test = X_test['income']
X_test = X_test.drop(['income'], axis=1)

# Converting to Values

X_train = X_train.values
X_test = X_test.values

# Print Shape of Training and Test sets

print(X_train.shape, X_test.shape)      # (19810, 9) (6513, 9)
print(y_train.shape, y_test.shape)      # (19810,) (6513,)

###########################################  Function Creation ###########################################











################################# Variables ##################################
#  Building Neuron Layers with 100, 50, 50 and 100 respectively 

n_cols = X_train.shape[1]                       # Number of Columns 
encoding_dim = 100

###########################################  Keras Code ######################################

# Input Shape to use the first hidden layer
input_layer = Input(shape = (n_cols, ))

# Build the model

model = Sequential()

#Creating the first hidden layer  (Encoder)

# model.add(Dense(100, activation = 'relu', input_shape = (n_cols, )))

encoder = Dense(encoding_dim, activation = 'relu')(input_layer)
encoder1 = Dense(50, activation="relu")(encoder)

# Creating decoder

decoder = Dense(50, activation='relu')(encoder1)
decoder = Dense(n_cols, activation='relu')(decoder)

# Creating the output 
#Encode = Model(inputs=input_layer, outputs=encoder1)   
Autoencoder = Model(inputs=input_layer, outputs=decoder)   

# Building epochs and batch_size

nb_epoch = 100
batch_size = 256

# Compile the Model
Autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


# Fit the model 


checkpointer = ModelCheckpoint(filepath="model.h5", verbose=0, save_best_only=True)

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,write_images=True)

# Implies X_train is both the input and output, which is required for reconstruction 
 
history = Autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history
 
# Saving the model

Autoencoder = load_model('model.h5')


# Plot of the Model

epochs = range(nb_epoch)
plt.figure()
plt.plot(epochs, history['loss'], 'b', label = 'Training Loss')
plt.plot(epochs, history['val_loss'], 'r', label = 'Validation Loss')
plt.title('Training and Validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training Loss', 'Test Loss']) 
plt.show()



                         
 # Make Predictions

predictions = Autoencoder.predict(X_test)

# Mean Squared Error (MSE)

mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse, 'true_class': y_test})   


error_df.describe()       

 
## Plot ROC Curve
fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();
#
### Plot Confusion Matrix
##
#precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
#plt.plot(recall, precision, 'b', label='Precision-Recall curve')
#plt.title('Recall vs Precision')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.show()
           
             

# Select the Encoder half (of the AutoEncoder) for ClassificationEncoder for Full Classification

num_classes = 2         

# Applying Logistic Regression

Encode = Model(inputs=input_layer, outputs=encoder1)   
Ouput = Dense(num_classes, activation='softmax', kernel_regularizer=L1L2(l1=0.0, l2=0.1))(Encode.output)
Encoder_Class = Model(Encode.input,Ouput)
#Encoder_Class.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
#Encoder_Class.fit(X_train, y_train, epochs=100, batch_size=128, shuffle=True, validation_data=(X_test, y_test))

# Ignore 
   
## Compile the Encoder Model
#    
#num_classes = 2         
#
## Applying Logistic Regression
#out2 = Dense(num_classes, activation='softmax', kernel_regularizer=L1L2(l1=0.0, l2=0.1))(Encode.output)
#Encoder_Class = Model(Encode.input,out2)


#  Freezing the weights of the Encoder model


for l1, l2 in zip(Encode.layers[:2],Autoencoder.layers[0:2]):
    l1.set_weights(l2.get_weights())
    
    
# Get weights for Autoencoder 

Autoencoder.get_weights()[0][1]  

# Get weights for Endoder layer
Encode.get_weights()[0][1] 


# Only Training the fully connected part (by Freezing the Autoencoder weights)

for layer in Encode.layers[0:2]:
    layer.trainable = False


## Ignore this

#scores = Encoder_Class.evaluate(X_test, y_test, verbose=1) 
#print("Accuracy: ", scores[0], scores[1])


# Compile the Classification Model
    
Encoder_Class.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 


# Model Summary
    
    
Encoder_Class.summary()  

# Training the Classification Model

Encoder_Class.fit(X_train, y_train, epochs=100, batch_size=128, shuffle=True, validation_data=(X_test, y_test))


#Saving the Model

Encoder_Class.save_weights('Classification_complete.h5')  

      
# Re-training the model by making trainable layer to True

for layer in Encode.layers[0:2]:
    layer.trainable = True
    
# Re-compiling the model after re-training 

Encoder_Class.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
 
# Re-Training the Classification Model

Classification = Encoder_Class.fit(X_train, y_train, epochs=100, batch_size=128, shuffle=True, validation_data=(X_test, y_test),callbacks=[checkpointer, tensorboard]).history
#  Plot the loss between training and Validation data

#Accuracy = Classification.history['acc']
#Val_accuracy = Classification.history['val_acc']
#loss = Classification.history['loss']
#val_loss = Classification.history['val_loss']
#epochs = range(len(Accuracy))
#plt.plot(epochs, Accuracy, 'bo', label='Training accuracy')
#plt.plot(epochs, Val_accuracy, 'b', label='Validation accuracy')
#plt.title('Training and validation accuracy')
#plt.legend()
#plt.figure()
#plt.plot(epochs, loss, 'bo', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
#plt.title('Training and validation loss')
#plt.legend()
#plt.show() 

# Alternative way to show and tell

Accuracy = Classification['acc']
Val_accuracy = Classification['val_acc']
loss = Classification['loss']
val_loss = Classification['val_loss']
epochs = range(len(Accuracy))
plt.plot(epochs, Accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, Val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show() 
                
# Model Evaluation on Test Set

Test_eval = Encoder_Class.evaluate(X_test, y_test, verbose = 0)
print('Test loss:', Test_eval[0])
print('Test accuracy:', Test_eval[1])


# Predict Label

#predicted_classes = np.squeeze(Encoder_Class.predict(X_test))
#
#correct = np.where(predicted_classes==y_train)[0]
#print("Found %d correct labels" % len(correct))
#for i, correct in enumerate(correct[:1]):
#    plt.subplot(3,3,i+1)
#  #  plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
#    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_train[correct]))
#    plt.tight_layout()

 
threshold = 20000
#classification_report(test_labels[0:5], predicted_classes > threshold)

#from sklearn.metrics import classification_report
## target_names = ["Class {}".format(i) for i in range(num_classes)]
## target_names = ['class 0', 'class 1']
#print(classification_report(y_test, predicted_classes > threshold))
                   
LABELS = ["<=50k", ">50K"]                                  
predicted_classes = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, predicted_classes)
plt.figure(figsize=(3, 3))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Classification for Full Class

from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]   ## target_names = ['class 0', 'class 1']
print(classification_report(y_test, predicted_classes, target_names = target_names))


#  Creating Hyperparameter Tuning
def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return {"batch_size": batches, "optimizer": optimizers, "keep_prob": dropout}



#model = KerasClassifier(Classification, verbose=0)

hyperparameters = create_hyperparameters()

search = RandomizedSearchCV(estimator=model,param_distributions=hyperparameters, n_iter=10, n_jobs=1, cv=3, verbose=1)


search.fit(X_train, y_train, scoring = None)
print(search.best_params_)