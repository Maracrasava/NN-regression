import pathlib
import matplotlib as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto"
                                                     "-mpg/auto-mpg.data")
print(dataset_path)

header = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
dataset = pd.read_csv(dataset_path, names=header, na_values="?", comment='\t', sep=" ", skipinitialspace=True)

dataset.tail()

# See how many missing values are there
print(dataset.isna().sum())
# Remove missing values
dataset.dropna()

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
# print(dataset.tail(15))
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')  # Converting to one-hot
# print(dataset.tail(15))

# Splitting data

train_dataset = dataset.sample(frac=0.8, random_state=42)
test_dataset = dataset.drop(train_dataset.index)

# Splitting the feature that your machine will predict

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Calculating statistics about dataset
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
print(train_stats)


# Normalizing the data
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# Building, compiling the model
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()

# Training the model

print(train_labels)  # Predict MPG
print(normed_train_data)  # based on this

history = model.fit(
  normed_train_data, train_labels,
  epochs=1000, validation_split=0.2, verbose=0)

for i in range(318):
    print(f"Predict {model.predict(normed_test_data).flatten()}, Actual : {test_labels}")
