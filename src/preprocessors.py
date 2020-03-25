import numpy as np
from . import image_processing


def flip_v(img=None, action=None, sensor_data=None):
    if img is not None:
        img = image_processing.flip_img(img)
    if action is not None:
        action = action.copy()
        action[0] *= -1                       # Reverse steering sign
    if sensor_data is not None:
        sensor_data = sensor_data.copy()
        sensor_data = sensor_data[::-1]       # Reverse sensor direction
    return img, action, sensor_data


def blur(img, action=None, sensor_data=None):
    kernel_size = np.random.choice((3, 5))
    img = image_processing.blur_img(img, ksize=(kernel_size, kernel_size))
    return img, action, sensor_data


def illuminate(img, action=None, sensor_data=None):
    alpha = 1
    beta = 10*np.random.randint(-8, 8)
    img = image_processing.illuminate_img(img, alpha=alpha, beta=beta)
    return img, action, sensor_data


def get_preprocessor(name):
    preprocessor = globals()[name]
    return preprocessor
