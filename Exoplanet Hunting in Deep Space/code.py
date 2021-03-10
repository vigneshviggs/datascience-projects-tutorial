# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 17:38:01 2021

@author: vigne
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, BatchNormalization, Input, concatenate, Activation
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from scipy.ndimage.filters import uniform_filter1d

train_df = pd.read_csv('exoTrain.csv')
test_df = pd.read_csv('exoTest.csv')
print(train_df.shape)
print(test_df.shape)

x_train = train_df.iloc[:,1:]
y_train = train_df["LABEL"] - 1
print(x_train)

x_test = test_df.iloc[:,1:]
y_test = test_df["LABEL"] - 1

min_max_scaler = preprocessing.MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
print(x_train)
print(x_train.shape)
a=x_train[0,:]


x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)

xx_train = np.stack([x_train, uniform_filter1d(x_train, axis=1, size=200)], axis=2)
xx_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, size=200)], axis=2) 

hist = LinearRegression().fit(x_train, y_train)
hist.predict(x_test)


model = Sequential()
model.add(Conv1D(input_shape=(3197, 2), filters=8, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=16, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=64, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=4))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(1e-5), loss = 'binary_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train, 
                           verbose=0, epochs=5)

plt.plot(hist.history['loss'], color='b')
#plt.plot(hist.history['val_loss'], color='r')
plt.show()
plt.plot(hist.history['accuracy'], color='b')
#plt.plot(hist.history['val_acc'], color='r')
plt.show()



