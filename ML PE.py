# -*- coding: utf-8 -*-
"""

@author: Mareike Thurau
@author: Silvan Mueller

"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from random import random

# einlesen der Daten
trainingData = pd.read_csv("../data_u_1_3001879_5279494.csv",sep=';')
# trainingData['Einordnung'] = pd.Categorical(trainingData['Einordnung'])
# trainingData['Einordnung'] = trainingData.Einordnung.cat.codes

target = trainingData.pop('Einordnung')

dataset = tf.data.Dataset.from_tensor_slices((trainingData.values, target.values))

print(dataset)
"""
class determineWeightCategory():

    def __init__ (self): 
        model = keras.Sequential([
            keras.layers.Flatten(input_shape(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
"""        