#!/usr/bin/env python3

import numpy as np

import tensorflow as tf

x_values = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_values = np.array([[0.0], [1.0], [1.0], [0.0]])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4, activation='tanh', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='tanh'),
])

model.compile(
    optimizer='adam',
    loss='MeanSquaredError',
    metrics=['accuracy'])

model.fit(x_values, y_values, epochs=2**12)
model.evaluate(x_values, y_values)

