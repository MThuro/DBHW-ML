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

# Create preprocessors
labelEncoder = preprocessing.LabelEncoder()
scaler = preprocessing.StandardScaler()

# Request file path from user
data_file = input("Enter the path of your data file: ")

# Read data from csv file
dataframe = pd.read_csv(data_file,sep=';')

# Convert column "Einordnung" into a discrete numerical value
dataframe['Einordnung'] = pd.Categorical(dataframe['Einordnung'])
dataframe['Einordnung'] = dataframe.Einordnung.cat.codes

# Standardize numeric columns in order to increase accuracy
dataframe[['Groesse', 'Gewicht', 'Alter']] = scaler.fit_transform(dataframe[['Groesse', 'Gewicht', 'Alter']])

# Split data into train, test and validation examples
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

# Create a tf.data dataset from the dataframe
def df_to_dataset(features: np.ndarray, labels: np.ndarray, shuffle=True, batch_size=80):
    labels = labelEncoder.fit_transform(labels)
    ds = tf.data.Dataset.from_tensor_slices(
        ({"feature": features}, {"Einordnung": labels}))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features))
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
def define_model(input_shape: tuple, neuron_layer_1: int = 128, neuron_layer_2: int = 128) -> tf.keras.Model:
      
    inputs = tf.keras.Input(shape=input_shape, name="feature")
    x = tf.keras.layers.Dense(neuron_layer_1, activation="relu", name="hidden_layer_1")(inputs)
    x = tf.keras.layers.Dense(neuron_layer_2, activation="relu", name="hidden_layer_2")(x)
    categories = tf.keras.layers.Dense(3, activation="softmax", name="Einordnung")(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=categories, name="keras/tf_categories")
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  run_eagerly=True)
    return model                    

# Train model
def train_model():
      best_test_acc = 0
      best_model = None

      # Get features from dataframes
      feature_convert = create_feature_layer()
      train_features = feature_convert(dict(train)).numpy()
      val_features = feature_convert(dict(val)).numpy()
      test_features = feature_convert(dict(test)).numpy()

      # Build data set
      train_ds = df_to_dataset(train_features, train["Einordnung"].values)
      val_ds = df_to_dataset(val_features, val["Einordnung"].values, shuffle=False)
      test_ds = df_to_dataset(test_features, test["Einordnung"].values, shuffle=False)

      # Get model
      for x in range (0,3):
        neuron_layer_1 = np.random.randint(50, 150)
        neuron_layer_2 = np.random.randint(50, 150)

        model = define_model((train_features.shape[1],), neuron_layer_1, neuron_layer_2)
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
      
# Execute training
test_accuracy, best_model = train_model()
print(test_accuracy)
# Save model
saved_model_path = "models/" + \
    f"{test_accuracy:.2f}"+"_"+time.strftime("%d%m%Y%H%M%S")+".h5"
best_model.save(saved_model_path)
print("File has been saved under", saved_model_path)