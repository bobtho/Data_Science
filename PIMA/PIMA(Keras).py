
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Analysis of PIMA Indian Diabetes dataset using Keras

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


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
from time import time 

from keras.models import Model,Sequential, load_model
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier

from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras.utils import to_categorical

# Creating a PIMA Dataframe

PIMA = pd.read_csv(r'C:\Users\Bobby\Desktop\AUT Datasets\PIMA_Indian_Diabetes.csv') 

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

# Print Class values count of Target variable (Highly Imbalanced data)
# There were 492 instances of Fraud (Class 1) and 284315 instances of Non Fraudulent (Class 0) 

PIMA_count = PIMA['Class'].value_counts() 
print(PIMA_count)

# Plot of Class variable distribution

fig = sns.barplot(x = [0, 1], y = PIMA_count, data = PIMA, color = 'blue')
plt.ylabel('Frequency of Class variable')
plt.xlabel ('Class variable')
plt.show(fig)

#################################################### Creating Train/Test Split Data ######################################

# Print Class values count of Target variable (Highly Imbalanced data)
# Class 0 means (Non-Diabetic) and Class 1 (Diabetic) 
# and ---> 500 (Non-Diabetic) and 268 (Diabetic)

X_train, X_test = train_test_split(PIMA, test_size = 0.20, random_state = 123)

# Create Training Set on Non-Diabetic Class (Class 0)

X_train = X_train[X_train.Class == 0]
y_train = X_train['Class']
X_train = X_train.drop(['Class'], axis=1)

# Create Test set on Diabetic Class (Class 1)

y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)

# Converting to Values

X_train = X_train.values
X_test = X_test.values

# Print Shape of Training and Test sets

print(X_train.shape, X_test.shape)      # (404, 8) (154, 8)
print(y_train.shape, y_test.shape)      # (404,) (154,)


################################# Variables ##################################
#  Building Neuron Layers with 100, 50, 50 and 100 respectively 

n_cols = X_train.shape[1]                       # Number of Columns 
encoding_dim = 100                              # Number of Neurons (Nodes) (in Input Layers) 

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


# Fit the model (Training the model)

checkpointer = ModelCheckpoint(filepath="model.h5", verbose=0, save_best_only=True)

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,write_images=True)

# Implies X_train is both the input and output, which is required for reconstruction 

start = time() 
history = Autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history

end = time()
print ("Trained model in {:.4f} seconds".format(end - start))         # Trained model in 10.7568 seconds
                          
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

# Show the Reconstruction Error
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


#  Precision Vs Recall curve

precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
plt.plot(recall, precision, 'r', label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()           

################################# Creating Encoder Classificaton ##########################################

# Select the Encoder half (of the AutoEncoder) for Full Classification

num_classes = 2       # The number of binary classes in the model        

# Creating an fully connected Encoder model to apply Classification
# Will apply Softmax classifier to the output. 

Encode = Model(inputs=input_layer, outputs=encoder1)   
Ouput = Dense(num_classes, activation='softmax')(Encode.output)
Encoder_Class = Model(Encode.input,Ouput)


# As the Autocencoder (Encoder + Decoder model) model was fully trained above, 
# lines (121 - 170), we do not require to train the Encoder again.  This part of 
# the code freezes the weights of the previously created Encoder model

for l1, l2 in zip(Encode.layers[:2],Autoencoder.layers[0:2]):
    l1.set_weights(l2.get_weights())
    
    
# Get weights for Autoencoder 
# This is to verify the new created Encoder has the same weights as the
# previously created Encoder component in the Autoencoder model
    
 # Get weights for Endoder layer in the newly created Encoder model   

Autoencoder.get_weights()[0][1]  

# Get weights for Endoder layer
Encode.get_weights()[0][1] 


# Only Training the fully connected part (by Freezing the Autoencoder weights)

for layer in Encode.layers[0:2]:
    layer.trainable = False

# Compile the Classification Model
    
Encoder_Class.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

# Model Summary
   
Encoder_Class.summary()  

# Training the Classification Model

start = time()
Encoder_Class.fit(X_train, y_train, epochs=100, batch_size=128, shuffle=True, validation_data=(X_test, y_test))

end = time()
print ("Trained model in {:.4f} seconds".format(end - start))       # Trained model in 2.1083 seconds

#Saving the Model

Encoder_Class.save_weights('Classification_complete.h5')  

      
# Re-training the model by making trainable layer to True

for layer in Encode.layers[0:2]:
    layer.trainable = True
    
# Re-compiling the model after re-training 


Encoder_Class.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
 
# Re-Training the Classification Model

start = time()

Classification = Encoder_Class.fit(X_train, y_train, epochs=100, batch_size=128, shuffle=True, validation_data=(X_test, y_test),callbacks=[checkpointer, tensorboard]).history

end = time()
print ("Trained model in {:.4f} seconds".format(end - start))

#  Plot the accuracy and loss between training and Validation data

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

# A random Threshold value above while transactions are labelled Diabetic
 
threshold = 50

# Confusion matrix Constructed on the same model as Sci-kit learn framework
                   
LABELS = ["Non-Diabetic", "Diabetic"]                                  
predicted_classes = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, predicted_classes)
plt.figure(figsize=(3, 3))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Classification for Full Class

target_names = ["Class {}".format(i) for i in range(num_classes)]   ## target_names = ['class 0', 'class 1']
print(classification_report(y_test, predicted_classes, target_names = target_names))

