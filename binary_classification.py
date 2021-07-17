# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 00:30:43 2021

@author: akino

 Binary Classification to predict whether a person has a heart disease or not.
"""

import numpy as np # for matrix operations
import pandas as pd # for loading CSV Files
import matplotlib.pyplot as plt # for Data Visualization

#loading the data
data = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/heart_disease.csv")

#splitting the dataset into input and output features
X = data.drop('target', axis=1) #Input variables
# axis=1 indicates that a column will be dropped
y = data['target'] # Target variable

#split the data into test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#building the model

import tensorflow as tf # Importing the TensorFlow Library
from tensorflow import keras # Import Keras from TensorFlow

from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

#firstly, we define/create the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
from tensorflow.keras.optimizers import RMSprop
optimizer = RMSprop(0.001) # Here, we have set our learning rate as 0.001
model.compile(loss='binary_crossentropy', optimizer= optimizer , metrics=['accuracy'])

print(model.summary())
# plotting the model
from tensorflow.keras.utils import plot_model
plot_model(model)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=10, verbose=1)
model.evaluate(X_test, y_test)

#plots
#Model Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'])
plt.show()

#Model Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'])
plt.show()
