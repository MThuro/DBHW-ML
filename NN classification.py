# -*- coding: utf-8 -*-
"""

@author: Mareike Thurau
@author: Silvan Mueller

"""
import pandas as pd
import numpy as np
import tensorflow as tf
import time
from tensorflow import keras
from tensorflow import feature_column
from sklearn.model_selection import train_test_split

from random import random

tf.enable_eager_execution() 

# Read data from csv file
dataframe = pd.read_csv("../data_u_1_3001879_5279494.csv",sep=';')
# Convert column "Einordnung" into a discrete numerical value
dataframe['Einordnung'] = pd.Categorical(dataframe['Einordnung'])
dataframe['Einordnung'] = dataframe.Einordnung.cat.codes

# Split data into train, test and validation examples
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

# Create a tf.data dataset from the dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=78):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Einordnung')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

def create_feature_layer() -> tf.keras.layers.DenseFeatures:
  # feature column for height
  feature_height = feature_column.numeric_column("Groesse")

  # feature column for weight
  feature_weight = feature_column.numeric_column("Gewicht")

  # feature column for age
  feature_age = feature_column.numeric_column("Alter") 

  # category column for gender
  feature_gender = feature_column.categorical_column_with_vocabulary_list(
        'Geschlecht', ['w', 'm'])
  feature_gender_one_hot = feature_column.indicator_column(feature_gender)

  # category column for activities  
  feature_activities = feature_column.categorical_column_with_vocabulary_list(
         'Betaetigung', ['keinSport', 'Kraftsport', 'Ausdauersport'])
  feature_activities_one_hot = feature_column.indicator_column(feature_activities)

  feature_columns = [feature_height, feature_weight,
                     feature_age, feature_gender_one_hot, feature_activities_one_hot]

  return tf.keras.layers.DenseFeatures(feature_columns)

# define model
def define_model(neuron_layer_1: int = 128, neuron_layer_2: int = 128) -> tf.keras.Sequential:
  #get feature layer
  feature_layer = create_feature_layer()

  model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(neuron_layer_1, activation='relu'),    
    tf.keras.layers.Dense(neuron_layer_2, activation='relu'),    
    tf.keras.layers.Dense(3, activation='softmax')
  ])

  model.compile(optimizer="adam",
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
                run_eagerly=True)
  return model                           

# train model
def train_model():
      # build data set
      train_ds = df_to_dataset(train)
      val_ds = df_to_dataset(val, shuffle=False)
      test_ds = df_to_dataset(test, shuffle=False)

      # get model
      model = define_model(111, 113)
      model.fit(train_ds,
                validation_data=val_ds,
                epochs=5)

      # evaluate model on test data to see accuracy
      test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
      # evaluate model on train data to crosscheck with tets data evaluation --> avoid overfitting
      train_loss, train_accuracy = model.evaluate(train_ds, verbose=0)
      return test_accuracy, model

#predict classification for test data
def predict(model: tf.keras.Sequential):
  test_ds = df_to_dataset(dataframe, shuffle=False)
  predictions = model.predict(test_ds)    

  # print results
  result = pd.DataFrame(predictions, columns=["Normalgewicht", "Uebergewicht", "Untergewicht"])
  result['Normalgewicht'] = pd.Series(["{0:.2f}%".format(val * 100) for val in result['Normalgewicht']], index = result.index)
  result['Übergewicht'] = pd.Series(["{0:.2f}%".format(val * 100) for val in result['Übergewicht']], index = result.index)
  result['Untergewicht'] = pd.Series(["{0:.2f}%".format(val * 100) for val in result['Untergewicht']], index = result.index)

  # save to csv
  file_name = "./predictions" + time.strftime("%d%m%Y%H%M%S") + ".csv"
  result.to_csv(file_name, sep=';', encoding='utf-8')     
      
#execute training
test_accuracy, model = train_model()
predict(model)
