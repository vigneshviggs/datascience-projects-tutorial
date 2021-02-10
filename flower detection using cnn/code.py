# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 16:26:36 2021

@author: vigne
"""


#%%
"""
Importing libraries:

Matplotlib for data visualization.
PIL for support for opening, manipulating, and saving many different image file formats.
os provides functions for interacting with the operating system.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
#%%
"""
The dataset contains about 3,700 photos of flowers. The dataset contains 5 sub-directories, one per class:

flower_photo/
  daisy/
  dandelion/
  roses/
  sunflowers/
  tulips/
"""

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

"To see the roses, use .glob to make a list of roses and access using .open in PIL"
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))
#%%
"""
Loading the images off disk using image_dataset_from_directory utility.
It will convert to td.data.Dataset for us. 
"""

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size = batch_size)





























