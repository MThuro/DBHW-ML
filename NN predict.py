from __future__ import absolute_import, division, print_function, unicode_literals

# import the necessary packages
import tensorflow as tf
from tensorflow import feature_column
import numpy as np
import argparse
import random
import h5py
import time
import sys

from sklearn import preprocessing
import numpy as np
import pandas as pd
from keras.models import load_model

tf.enable_eager_execution()

# Request file paths
data_file = input("Enter the path of your data file: ")
model_file = input("Enter the path of your model file: ")

# Create preprocessors
labelEncoder = preprocessing.LabelEncoder()
scaler = preprocessing.StandardScaler()

# Create dataframe and labels
def df_to_dataset():
    # Read data from csv file
    dataframe = pd.read_csv(data_file,sep=';')

    # Preprocess data
    dataframe['Einordnung'] = pd.Categorical(dataframe['Einordnung'])
    dataframe['Einordnung'] = dataframe.Einordnung.cat.codes
    dataframe[['Groesse', 'Gewicht', 'Alter']] = scaler.fit_transform(dataframe[['Groesse', 'Gewicht', 'Alter']])

    return dataframe

# Create feature layer
def create_feature_layer() -> tf.keras.layers.DenseFeatures:
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

# Load Model from file
def loadModel():
    model_from_saved = tf.keras.models.load_model(model_file)
    model_from_saved.summary()
    return model_from_saved

# Predict classification for test data
def predict(model: tf.keras.Model, dataframe):
  # Create feature mapping for dataframe
  feature_convert = create_feature_layer()
  data_features = feature_convert(dict(dataframe)).numpy()

  # Predict with learned model
  predictions = model.predict({"feature": data_features})    

  # Process results
  result = pd.DataFrame(predictions, columns=["Normalgewicht", "Übergewicht", "Untergewicht"])
  result['Normalgewicht'] = pd.Series(["{0:.2f}%".format(val * 100) for val in result['Normalgewicht']], index = result.index)
  result['Übergewicht'] = pd.Series(["{0:.2f}%".format(val * 100) for val in result['Übergewicht']], index = result.index)
  result['Untergewicht'] = pd.Series(["{0:.2f}%".format(val * 100) for val in result['Untergewicht']], index = result.index)

  # Save to csv
  file_name = "./predictions" + time.strftime("%d%m%Y%H%M%S") + ".csv"
  result.to_csv(file_name, sep=';', encoding='utf-8')     
  print("File has been saved under", file_name)

# Load and preprocess data from provided file
dataframe = df_to_dataset()
saved_model = loadModel()
predict(saved_model, dataframe)