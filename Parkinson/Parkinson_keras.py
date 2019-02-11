# -*- coding: utf-8 -*-
"""

Analysis of Parkinson dataset using Keras (HyperTuned)

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

# Creating a Parkinson Dataframe

#Parkinson = pd.read_csv(r'H:\AUT Datasets\Parkinson.csv')                              # AUT Destination
#Parkinson = pd.read_csv(r'C:\Users\user\Desktop\AUT Datasets\Parkinson.csv')
Parkinson = pd.read_csv(r'C:\Users\Bobby\Desktop\AUT Datasets\LSVT_Voice.csv')  
print(Parkinson)

# Explore the Data

Parkinson.info()                          # 284807 (Rows) and 31 (Columns)

# Describe the Data

Parkinson.describe()
Parkinson.columns

#Code to check for shape of data
 
print ("Number of rows:  ", Parkinson.shape[0])   # Gives the number of (Rows) only  284807 Instances 
print ("Number of rows:  ", Parkinson.shape[1])   # Number of Columns (31)
print ("Number of rows:  ", Parkinson.shape)      #284807 Instances (Rows)  and 31 Features (Columns) 

#Code to check for any missing values

Parkinson.isnull().any()
Parkinson.isnull().values.any()
Parkinson.isnull().sum()  

# Print Class values count of Target variable (Highly Imbalanced data)
# There were 492 instances of Fraud (Class 1) and 284315 instances of Non Fraudulent (Class 0) 

Parkinson_count = Parkinson['Class'].value_counts() 
print(Parkinson_count)


Parkinson = Parkinson.drop(['Subject_index', 'Age', 'Gender'], axis=1)
Parkinson.Class.replace([1, 2], [0, 1], inplace=True)                            # Replace Class 1 --> 0 and Class 2 ---> 1     
Parkinson_count = Parkinson['Class'].value_counts() 
print(Parkinson_count)

# Plot of Class variable distribution

fig = sns.barplot(x = [0, 1], y = Parkinson_count, data = Parkinson, color = 'blue')
plt.ylabel('Frequency of Class variable')
plt.xlabel ('Class variable')
plt.show(fig)

# Creating a new Column with Normalised Amount and removing the Amount and Time Column in DataFrame

#################################################### Creating Train/Test Split Data ######################################

# Create 80/20 (Train/Test) split with Random State Values across both the classes (Fraud and Non-Fraud)
# Creating Input Features (X) and Target variable (y)

X_train, X_test = train_test_split(Parkinson, test_size = 0.20, random_state = 123)

# Create Training Set on Unacceptable Class (Class 1)

X_train = X_train[X_train.Class == 1]
y_train = X_train['Class']
X_train = X_train.drop(['Class'], axis=1)

# Create Test set on Non-Fraud Class (Class 0)

y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)

# Converting to Values

X_train = X_train.values
X_test = X_test.values

# Print Shape of Training and Test sets

print(X_train.shape, X_test.shape)      # (67, 310) (26, 310)
print(y_train.shape, y_test.shape)      # (67,) (26,)

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
batch_size = 64

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

 
7


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