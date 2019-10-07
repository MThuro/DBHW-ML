from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import time
import sys
import math

import tensorflow as tf

from tensorflow import feature_column
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

labelEncoder = preprocessing.LabelEncoder()
tf.enable_eager_execution()

#if len(sys.argv) < 2:
#    print("Please provide data file to learn")
#    exit()

dataframe = pd.read_csv("../data_u_1_3001879_5279494.csv", sep=";")
dataframe.head()

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


def create_dataset(train_features, val_features, test_features, batch_size=32):
    train_ds = df_to_dataset(
        train_features, train["Einordnung"].values, batch_size=batch_size)
    val_ds = df_to_dataset(
        val_features, val["Einordnung"].values, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(
        test_features, test["Einordnung"].values, shuffle=False, batch_size=batch_size)
    return train_ds, val_ds, test_ds


def df_to_dataset(features: np.ndarray, labels: np.ndarray, shuffle=True, batch_size=32) -> tf.data.Dataset:
    labels = labelEncoder.fit_transform(labels)
    ds = tf.data.Dataset.from_tensor_slices(
        ({"feature": features}, {"Einordnung": labels}))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features))
    ds = ds.batch(batch_size)
    return ds


def sortAccuracy(val):
    return val[0]


def create_feature_layer() -> tf.keras.layers.DenseFeatures:
    # feature column for height
    feature_height = feature_column.numeric_column("Groesse")

    # feature column for weight
    feature_weight = feature_column.numeric_column("Gewicht")

    # bucketized column for age -> buckets by BMI classification
    #numeric_feature_age = feature_column.numeric_column("Alter")
    feature_age = feature_column.numeric_column(
        "Alter")  # feature_column.bucketized_column(
    # numeric_feature_age, boundaries=[15, 25, 50, 75, 85, 90, 95])

    # category column for gender
    feature_gender = feature_column.categorical_column_with_vocabulary_list(
        'Geschlecht', ['w', 'm'])
    feature_gender_one_hot = feature_column.indicator_column(feature_gender)

    # category column for sports activities
    feature_sports = feature_column.categorical_column_with_vocabulary_list(
        'Betaetigung', ['keinSport', 'Kraftsport', 'Ausdauersport'])
    feature_sports_one_hot = feature_column.indicator_column(feature_sports)

    feature_columns = [feature_height, feature_weight,
                       feature_age, feature_gender_one_hot, feature_sports_one_hot]

    return tf.keras.layers.DenseFeatures(feature_columns)


def get_model(input_shape: tuple, learning_rate: float = 0.01, neuron_1: int = 128, neuron_2: int = 128) -> tf.keras.Model:
    # Construct the model

    inputs = tf.keras.Input(shape=input_shape, name="feature")
    x = tf.keras.layers.Dense(neuron_1, kernel_initializer="normal", activation="relu", name="hidden_layer_1")(inputs)
    x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x) #half/third of processing time but: lower accuracy
    x = tf.keras.layers.Dense(neuron_2, kernel_initializer="normal", activation="relu", name="hidden_layer_2")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x) #half/third of processing time but: lower accuracy
    categ_pred = tf.keras.layers.Dense(3, activation="sigmoid", name="Einordnung")(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=categ_pred, name="keras/tf_categ")
    model.compile(optimizer="adam",  # tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  run_eagerly=True)
    return model


BATCH_COUNT = 3
NEURON_1_COUNT = 3
NEURON_2_COUNT = 3

batch_sizes = np.random.randint(30, 100, [BATCH_COUNT])
neuron_1_array = np.random.randint(50, 150, [NEURON_1_COUNT])
neuron_2_array = np.random.randint(50, 150, [NEURON_2_COUNT])

result = []
counter = 0
total = BATCH_COUNT * NEURON_1_COUNT * NEURON_2_COUNT
totalruntime = 0


def train_model():
    global counter
    global totalruntime

    top_accuracy = 0
    top_model = None

    # Transform all features into dense tensors.
    feature_transform = create_feature_layer()
    train_features = feature_transform(dict(train)).numpy()
    val_features = feature_transform(dict(val)).numpy()
    test_features = feature_transform(dict(test)).numpy()

    # loop for batch size
    for batch in batch_sizes:
        train_ds, val_ds, test_ds = create_dataset(
            train_features, val_features, test_features, batch)

        # loop for neuron count in hidden layer 1
        for neuron_1 in neuron_1_array:
            # loop for neuron count in hidden layer 2
            for neuron_2 in neuron_2_array:
                # store start time
                start = time.time()

                # define model with neurons from neuron arrays
                model = get_model(input_shape=(
                    train_features.shape[1],), learning_rate=0.01, neuron_1=neuron_1, neuron_2=neuron_2)

                # do model training
                model.fit(train_ds,
                          validation_data=val_ds,
                          epochs=5,
                          verbose=0)

                # evaluate model on test data to see accuracy
                test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
                # evaluate model on train data to crosscheck with tets data evaluation --> avoid overfitting
                train_loss, train_accuracy = model.evaluate(train_ds, verbose=0)

                # calculate runtime
                end = time.time()
                runtime = end - start
                totalruntime = totalruntime + runtime
                runtimeString = ""
                if runtime > 60:
                    runtimeString = "> 1 min"
                else:
                    runtimeString = f"{runtime:.2f}s"

                counter = counter + 1

                # check if new top model is found
                if test_accuracy > top_accuracy:
                    top_accuracy = test_accuracy
                    top_model = model
                    print("********************************")
                    print("New Top accuracy:", f"{top_accuracy*100:.2f}%")
                    print("********************************")

                print(f"{(counter / total)*100:.2f}% Processed")

                # build result set
                result.append([test_accuracy, batch, neuron_1,
                               neuron_2, train_accuracy, runtimeString])
    return top_accuracy, top_model


top_acc, top_model = train_model()

print("Top:", top_acc)

result.sort(key=sortAccuracy, reverse=True)

# create output
df = pd.DataFrame(result, columns=["Test Accuracy", "Batch size",
                                   "Neurons Layer 1", "Neurons Layer 2", "Training Accuracy", "Runtime"])

saved_model_path = "saved_models/" + \
    f"{top_acc:.2f}"+"_"+str(result[0][1])+"_" + \
    str(result[0][2])+"_"+str(result[0][3])+".h5"
top_model.save(saved_model_path)
#saved_model_path_tf = "tf_"+saved_model_path
#tf.saved_model.save(top_model, saved_model_path_tf)

print(df)
print("Model saved:", saved_model_path)

# calculate runtime
mins = math.floor(totalruntime / 60)
secs = totalruntime % 60
print("Total runtime:", f"{mins}min", f"{secs:.2f}s")