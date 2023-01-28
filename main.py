#!/usr/bin/env python3

import numpy as np
from PIL import Image

import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = list(x_train)
y_train = list(y_train)

for i in range(len(y_train)):
    one_hot = np.zeros(11)
    one_hot[y_train[i]] = 1.0
    y_train[i] = one_hot
y_test = list(y_test)
for i in range(len(y_test)):
    one_hot = np.zeros(11)
    one_hot[y_test[i]] = 1.0
    y_test[i] = one_hot


for _ in range(2**18):
    x_train.append(np.random.random((28, 28)))
    one_hot = np.zeros(11)
    one_hot[10] = 1.0
    y_train.append(one_hot)
temp = list(zip(x_train, y_train))
np.random.shuffle(temp)
x_train, y_train = zip(*temp)


x_train = np.array(x_train)
y_train = np.array(y_train)
y_test = np.array(y_test)


x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(11, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

def get_best_input_for_model(model, starter_input, expected_output, ranges):
    shape = (2**8, 28, 28)
    # inputs = (np.random.random(shape)**2-np.random.random(shape)**2) / 10.0
    inputs = (np.random.random(shape)-0.5)**5

    for i in range(len(inputs)):
        inputs[i] += starter_input
        for j in range(len(inputs[i])):
            for k in range(len(inputs[i][j])):
                if inputs[i][j][k] < ranges[0]:
                    inputs[i][j][k] = ranges[0]
                if inputs[i][j][k] > ranges[1]:
                    inputs[i][j][k] = ranges[1]

    outputs = model.predict(inputs)

    best_input_index = 0
    best_input_score = None

    for i in range(len(inputs)):
        score = ((np.array(expected_output) - np.array(outputs[i]))**2).sum()
        if best_input_score == None or score < best_input_score:
            best_input_score = score
            best_input_index = i

    return inputs[best_input_index]
def get_image_model(model, starter_input, expected_output, times, ranges):
    for i in range(times):
        starter_input = get_best_input_for_model(
                model,
                starter_input,
                expected_output,
                ranges)

    return starter_input

def get_best_input_for_layer(layer, starter_input, expected_output, ranges):
    shape = (256, layer.input.shape[1])
    inputs = (np.random.random(shape)**2-np.random.random(shape)**2) / 10.0

    for i in range(len(inputs)):
        inputs[i] += starter_input
        for j in range(len(inputs[i])):
            if inputs[i][j] < ranges[0]:
                inputs[i][j] = ranges[0]
            if inputs[i][j] > ranges[1]:
                inputs[i][j] = ranges[1]

    outputs = layer(inputs)

    best_input_index = 0
    best_input_score = None

    for i in range(len(inputs)):
        score = ((np.array(expected_output) - np.array(outputs[i]))**2).sum()
        if best_input_score is None or score < best_input_score:
            best_input_score = score
            best_input_index = i

    return inputs[best_input_index]

def get_best_input_for_layer_times(layer, starter_input, expected_output, times, ranges):
    for _ in range(times):
        starter_input = get_best_input_for_layer(layer, starter_input, expected_output, ranges)
    return starter_input

def get_image(model, expected_output, layers, times, ranges):
    next_output = expected_output
    for i in range(len(layers)):
        next_output = get_best_input_for_layer_times(
                model.layers[layers[i]],
                np.zeros(model.layers[layers[i]].input.shape[1]), next_output, times[i], ranges[i])

    return next_output


def get_image_of(numbers, model, ranges):
    output = np.zeros(11)
    for x in numbers:
        output[x] = 1.0

    shape = model.layers[0].input.shape

    return get_image(model, output, [2,1], [2**7, 2**7], ranges)
    # return get_image_model(
    #     model,
    #     np.zeros((shape[1], shape[2])),
    #     output,
    #     512,
    #     (0.0, 1.0))


def to_image(image):
    # print(image)
    new_image = (image*255).astype('uint8')
    # print(new_image)
    img = Image.fromarray(new_image)

    p = model.predict(image.reshape(-1, 28, 28))
    print(p)
    print(np.argmax(p, axis=1))
    img.show()

# for i in range(10):
#     image = get_image_of([i], model)
#     image = image.reshape((28,28))
#     to_image(image)

def show_image(values):
    image = get_image_of(values, model, [(0, 1), (0, 1)])
    image = image.reshape((28, 28))
    to_image(image)

show_image([5])
show_image([10])

# image = np.random.random((28, 28))
# to_image(image)

# for i in range(5):
#     to_image(x_train[i])

