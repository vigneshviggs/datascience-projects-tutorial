# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
#Importing libraries
"""
Pandas for data preprocessing/transform
Sklearn for data preprocessing and splitting
Keras for model building
Matplotlib for data visualization
"""
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
#%%
'Converting data to data frame'

df = pd.read_csv('housepricingdata.csv')
print(df)
#%%
'Slicing and Dicing of data is not done via dataframe. Hence convert to arrays by using .values function'
 
data = df.values

X = data[:,0:10]
Y = data[:,10]

print(X)
print(Y)
#%%
"""
Normalization of data is to be done to bring the data to same level and exclude outliers
As Y is a Boolean variable, there is no need to normalize it.
"""

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

print(X_scale)
#%%
"""
Split the data in training, validating and testing.
Use 70% of data for train, 15% for val and 15% for train
"""

X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size = 0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5)
#%%
"""
Use Sequential of Keras to build a model. Remember Dense is a fully connected layer. 
32 is number of neurons. Relu is activation unit, input shape is shape of X. 
At the last, use 1 layer(the layer of output variable Y) and activation as sigmoid(beneficial at the last layers. Normally we would not want...
                                                                                   sigmoid in the middle layers).
SGD is optimizer. binary_crossentropy is loss function as our outputs are 0/1. We want to track accuracy, so metrics=accuracy.
"""

model = Sequential([Dense(32, activation='relu', input_shape=(10,)), Dense(32, activation='relu'), Dense(1, activation='sigmoid'),])
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
#%%

hist = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))
#%%
"""
Function .evaluate shows the values in the order of loss and accuracy.
For showing only accuracy, use [1] behind, like:
    model.evaluate(X_test, Y_test)[1]
"""

model.evaluate(X_test, Y_test)
#%%
"""
Function:
    .plot plots the variables like loss, accuracy, etc
    .title is used for naming the figure
    .ylabel and .xlabel is used for labelling the axis
    .legend is used to show the color of the plots and represent the names of lines
    .show is used to show the figure
"""
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Epochs')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
#%%

'Example of L2 regularization and dropout by Overfitting the model'

"""
Here the data is overfitted first so that regularization and dropout can be applied. 
For overfitting, 1000 neurons layer is made with multiple layers like this.
Add kernel_regularizer=regularizers.l2(0.01) inside the layer(Dense) and add Dropout(0.3) at outside, after the Dense layer.
Here adam is used as optimizer. 
"""

from keras.layers import Dropout
from keras import regularizers

model_2 = Sequential([Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(10,)), Dropout(0.3), Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)), Dropout(0.3), Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)), Dropout(0.3), Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)), Dropout(0.3), Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),])
model_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist_2 = model_2.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))

model.evaluate(X_test, Y_test)

plt.plot(hist_2.history['loss'])
plt.plot(hist_2.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist_2.history['accuracy'])
plt.plot(hist_2.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Epochs')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()