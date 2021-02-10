# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:48:17 2021

@author: vigne
"""

#%%
"""
NOTE: Tensorboard versions >= 2.3.0 maybe supported.

Load the tensorboard notebook extension
"""

'Write this in Console/CMD'
%load_ext tensorboard
#%%

import tensorflow as tf
import datetime
from tensorboard.plugins.hparams import api as hp

'Clearing any logs from previous runs'

rm -rf ./logs/
#%%
"""
Loading and creating a model:

Download MNIST dataset from tf.keras
Load the train and test data using load_data()

Create a model using Sequential() which consists of following layers:
    Flatten with input shape of (28,28)
    Dense with 512 neurons and activation = relu
    Dropout by dropping 20% neurons
    Dense with 10 neurons or output units for classification and softmax activation
"""

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
#%%
"""
Creating logs for tensorboard

Firstly a model is created and then logs can be created during training by passing tensorboard and callbacks in Model.fit() of Keras
"""

model = create_model()
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

'creating logs at logs/fit location with date and time'
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

"""
Inserting logs in callbacks per each epoch
This can be done by using histogram_freq = 1
"""
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

'Inserting hyperparameters callback with specifying number of relu units and dropout'
hparams_callback = hp.KerasCallback(log_dir, {
    'num_relu_units': 512,
    'dropout': 0.2
})
#%%
"""
Train the model with 5 epochs and callbacks
"""

model.fit(
    x=x_train, 
    y=y_train, 
    epochs=5, 
    validation_data=(x_test, y_test), 
    callbacks=[tensorboard_callback, hparams_callback])
#%%
"""
Start TensorBoard through the command line or within a notebook
"""

"Type this in Console/CMD"
%tensorboard --logdir logs/fit 
%tensorboard — logdir=”./logs” — port 6006

#%%
import pkg_resources

for entry_point in pkg_resources.iter_entry_points('tensorboard_plugins'):
    print(entry_point.dist)
#%%

python -m tensorboard.main --logdir ./model --port 6006
