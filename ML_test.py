# -*- coding: utf-8 -*-
"""
@author: Mareike Thurau
@author: Silvan Mueller
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from random import random

#tf.enable_eager_execution() 

# Read data from csv file
dataframe = pd.read_csv("C:/Users/muellersm/Desktop/data_u_1_3001879_5279494.csv",sep=';')
# Convert column "Einordnung" into a discrete numerical value
dataframe['Einordnung'] = pd.Categorical(dataframe['Einordnung'])
dataframe['Einordnung'] = dataframe.Einordnung.cat.codes

# Convert column "Geschlecht" into a discrete numerical value
dataframe['Geschlecht'] = pd.Categorical(dataframe['Geschlecht'])
dataframe['Geschlecht'] = dataframe.Geschlecht.cat.codes

# Convert column "Betaetigung" into a discrete numerical value
dataframe['Betaetigung'] = pd.Categorical(dataframe['Betaetigung'])
dataframe['Betaetigung'] = dataframe.Betaetigung.cat.codes

# Split data into train, test and validation examples
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

# Create a tf.data dataset from the dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Einordnung')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

train_ds = df_to_dataset(train)
val_ds = df_to_dataset(val, shuffle=False)
test_ds = df_to_dataset(test, shuffle=False)

feature_columns = []

# Numeric columns
for header in ['Geschlecht', 'Groesse', 'Alter', 'Betaetigung']:
  feature_columns.append(tf.feature_column.numeric_column(header))


feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  tf.keras.layers.Dense(5, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),  
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),        
  tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

