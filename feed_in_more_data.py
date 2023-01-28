#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from PIL import Image

import functions as f

def add_in_data_set(x_train, y_train, addition_x, addition_y):
    x_new_shape = list(x_train.shape)
    x_new_shape[0] = -1
    x_new_shape = tuple(x_new_shape)

    y_new_shape = list(y_train.shape)
    y_new_shape[0] = -1
    y_new_shape = tuple(y_new_shape)

    new_x = np.append(x_train, addition_x).reshape(x_new_shape)
    new_y = np.append(y_train, addition_y).reshape(y_new_shape)

    temp = list(zip(new_x, new_y))
    np.random.shuffle(temp)
    new_x, new_y = zip(*temp)

    x_shape = list(x_train.shape)
    x_shape[0] = -1
    x_shape = tuple(x_shape)

    y_shape = list(y_train.shape)
    y_shape[0] = -1
    y_shape = tuple(y_shape)

    new_x = np.array(new_x).reshape(x_shape)
    new_y = np.array(new_y).reshape(y_shape)

    return (new_x, new_y)


if __name__ == '__main__':
    mnist = tf.keras.datasets.mnist

    num_labels = 11

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = np.squeeze(np.eye(num_labels)[y_train.reshape(-1)])
    y_test = np.squeeze(np.eye(num_labels)[y_test.reshape(-1)])

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_labels, activation='softmax')
    ])

    model.summary()

    model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


    x = np.random.random((1000, 28, 28))
    y = np.zeros((1000, num_labels))
    x, y = add_in_data_set(x_train, y_train, x, y)

    history = model.fit(x, y, epochs=2**2, validation_data=(x_test, y_test))

    image = np.zeros((28, 28))
    classify = 5
    for i in range(2**8):
        print('MAIN LOOP:', i, '/', 2**12)

        history = model.fit(x, y, epochs=1, validation_data=(x_test, y_test))

        new_x = []
        new_y = []
        for j in range(32):
            image = np.zeros((28, 28)) + 0.5 # reset image
            image = f.find_input(image, 2**8, 2**5, (0,1), np.eye(11)[classify], f.predict_model(model))
            count = 0
            while model.predict(image.reshape(-1, 28, 28))[0].argmax() != classify:
                image = f.find_input(image, 2**12, 2**6, (0,1), np.eye(11)[classify], f.predict_model(model))
                count += 1
                print('Could Not Come Up With Image')
                if count % 10 == 0:
                    f.to_image(image)
                    print(((f.show_prediction(model, image)-np.eye(11)[classify])**2).sum())

            new_x.append(image)
            new_y.append(np.eye(11)[10])
            if j == 31:
                f.to_image(image)
                f.show_prediction(model, image)
                #image = np.zeros((28, 28)) # reset image


        x, y = add_in_data_set(x, y, np.array(new_x), np.array(new_y))

