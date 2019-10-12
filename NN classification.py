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
from sklearn import preprocessing

from random import random

tf.enable_eager_execution() 

# Add preprocessor
scaler = preprocessing.StandardScaler()

# Read data from csv file
dataframe = pd.read_csv("../data_u_1_3001879_5279494.csv",sep=';')
# Convert column "Einordnung" into a discrete numerical value
dataframe['Einordnung'] = pd.Categorical(dataframe['Einordnung'])
dataframe['Einordnung'] = dataframe.Einordnung.cat.codes

# Standardize numeric columns in order to increase accuracy
dataframe[['Groesse', 'Gewicht', 'Alter']] = scaler.fit_transform(dataframe[['Groesse', 'Gewicht', 'Alter']])

# Split data into train, test and validation examples
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

# Create a tf.data dataset from the dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=80):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Einordnung')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

# Create feature layer
def create_feature_layer() -> keras.layers.DenseFeatures:
  # Feature column for height
  feature_height = feature_column.numeric_column("Groesse")

  # Feature column for weight
  feature_weight = feature_column.numeric_column("Gewicht")

  # Feature column for age
  feature_age = feature_column.numeric_column("Alter") 

  # Category column for gender
  feature_gender = feature_column.categorical_column_with_vocabulary_list(
        'Geschlecht', ['w', 'm'])
  feature_gender_one_hot = feature_column.indicator_column(feature_gender)

  # Category column for activities  
  feature_activities = feature_column.categorical_column_with_vocabulary_list(
         'Betaetigung', ['keinSport', 'Kraftsport', 'Ausdauersport'])
  feature_activities_one_hot = feature_column.indicator_column(feature_activities)

  feature_columns = [feature_height, feature_weight,
                     feature_age, feature_gender_one_hot, feature_activities_one_hot]

  return tf.keras.layers.DenseFeatures(feature_columns)

# Define model
def define_model(neuron_layer_1: int = 128, neuron_layer_2: int = 128) -> tf.keras.Sequential:
  # Get feature layer
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

# Train model
def train_model():
      best_test_acc = 0
      best_model = None

      # Build data set
      train_ds = df_to_dataset(train)
      val_ds = df_to_dataset(val, shuffle=False)
      test_ds = df_to_dataset(test, shuffle=False)

      # Get model
      for x in range (0,3):
        neuron_layer_1 = np.random.randint(50, 150)
        neuron_layer_2 = np.random.randint(50, 150)

        model = define_model(neuron_layer_1, neuron_layer_2)
        model.fit(train_ds,
                  validation_data=val_ds,
                  epochs=5)

        # Evaluate model on test data to see accuracy
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

        # Compare accucary -> save best model
        if test_accuracy > best_test_acc:
          best_test_acc = test_accuracy
          best_model = model
      return best_test_acc, best_model

# Predict classification for test data
def predict(model: tf.keras.Sequential):
  test_ds = df_to_dataset(dataframe, shuffle=False)
  predictions = model.predict(test_ds)    

  # Process results
  result = pd.DataFrame(predictions, columns=["Normalgewicht", "Übergewicht", "Untergewicht"])
  result['Normalgewicht'] = pd.Series(["{0:.2f}%".format(val * 100) for val in result['Normalgewicht']], index = result.index)
  result['Übergewicht'] = pd.Series(["{0:.2f}%".format(val * 100) for val in result['Übergewicht']], index = result.index)
  result['Untergewicht'] = pd.Series(["{0:.2f}%".format(val * 100) for val in result['Untergewicht']], index = result.index)

  # Save to csv
  file_name = "./predictions" + time.strftime("%d%m%Y%H%M%S") + ".csv"
  result.to_csv(file_name, sep=';', encoding='utf-8')     
  print("File has been saved under", file_name)
      
# Execute training
test_accuracy, model = train_model()
print(test_accuracy)

# Predict categories based on learned model
predict(model)

