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

from random import random

# Read data from csv file
dataframe = pd.read_csv("../data_u_1_3001879_5279494.csv",sep=';')
# Convert column "Einordnung" into a discrete numerical value
dataframe['Einordnung'] = pd.Categorical(dataframe['Einordnung'])
dataframe['Einordnung'] = dataframe.Einordnung.cat.codes

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
for header in ['Geschlecht', 'Groesse', 'Alter']:
  feature_columns.append(tf.feature_column.numeric_column(header))

# Indicator columns
geschlecht = tf.feature_column.categorical_column_with_vocabulary_list(
      'Geschlecht', ['m', 'w'])
geschlecht_one_hot = tf.feature_column.indicator_column(geschlecht)
feature_columns.append(geschlecht_one_hot)

betaetigung = tf.feature_column.categorical_column_with_vocabulary_list(
      'Betaetigung', ['keinSport', 'Kraftsport', 'Ausdauersport'])
betaetigung_one_hot = tf.feature_column.indicator_column(betaetigung)
feature_columns.append(betaetigung_one_hot)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
"""
model = tf.keras.Sequential([
  feature_layer,
  tf.compat.v1.layers.Dense(128, activation='relu'),
  tf.compat.v1.layers.Dense(128, activation='relu'),
  tf.compat.v1.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)
"""