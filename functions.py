#!/usr/bin/env python3

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

def find_input(starter_input, trials_per_time, times_update, ranges, expected_outputs, f):

    input_shape = starter_input.shape

    input_shape = list(input_shape)
    if input_shape[0] != -1:
        input_shape.insert(0, -1)
    input_shape[0] = trials_per_time-1
    input_shape = tuple(input_shape)

    for i in range(times_update):
        if i % 100 == 0 and times_update >= 1000:
            print(i, '/', times_update)


        random_inputs = list(np.random.random(input_shape))
        random_inputs.append(np.zeros(input_shape[1:]))
        random_inputs = np.array(random_inputs)

        inputs = np.clip(starter_input + ((random_inputs-0.5) ** 3), ranges[0], ranges[1])
        new_inputs = list(inputs)
        new_inputs.extend(blur_images(inputs))
        new_inputs.extend(round_images(inputs))
        new_inputs.extend(round_images(blur_images(inputs)))
        new_inputs.extend(round_images(blur_images(inputs, sigma=1)))
        new_inputs = np.array(new_inputs)
        inputs = new_inputs

        outputs = f(inputs)

        best_index = ((outputs-expected_outputs)**2).sum(axis=1).argmin()
        starter_input = inputs[best_index]

    return starter_input

# def find_input_from_model(model,
#                           trials_per_time,
#                           times_update,
#                           ranges,
#                           expected_outputs,
#                           score_output,
#                           f):
#     model.


def blur_images(images, sigma=3):
    return gaussian_filter(images, sigma=sigma)

def round_images(images):
    return np.round(images)


def to_image(image):
    new_image = (image*255).astype('uint8')
    img = Image.fromarray(new_image)
    img.show()

def show_prediction(model, image):
    p = model.predict(image.reshape(-1, 28, 28))
    print(p)
    print(np.argmax(p, axis=1))
    return p

def predict_model(model):
    def f(inputs):
        return model.predict(inputs)
    return f

