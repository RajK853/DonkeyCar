from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Cropping2D, Conv2D, Dense, MaxPool2D, Flatten, Dropout, Lambda
from .utils import clip_steering


DonkeyNetV0Layers = [Conv2D(filters=64, kernel_size=7, strides=2, activation="relu", kernel_regularizer=l2(0.001)),
                     MaxPool2D(pool_size=(4, 4), strides=2),
                     Conv2D(filters=32, kernel_size=5, strides=2, activation="relu", kernel_regularizer=l2(0.001)),
                     MaxPool2D(pool_size=(2, 2), strides=2),
                     Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", kernel_regularizer=l2(0.001)),
                     Flatten(),
                     Dense(units=64, activation="relu", kernel_regularizer=l2(0.001)),
                     Dropout(0.75),
                     Dense(units=32, activation="relu", kernel_regularizer=l2(0.001)),
                     Dense(units=1),
                     Lambda(clip_steering)]

DonkeyNetV1Layers = [Conv2D(filters=24, kernel_size=5, strides=2, activation="relu"),
                     Conv2D(filters=32, kernel_size=5, strides=2, activation="relu"),
                     Conv2D(filters=64, kernel_size=5, strides=2, activation="relu"),
                     Conv2D(filters=64, kernel_size=3, strides=1, activation="relu"),
                     Conv2D(filters=64, kernel_size=3, strides=1, activation="relu"),
                     Flatten(),
                     Dense(units=100, activation="relu"),
                     Dropout(0.1),
                     Dense(units=50, activation="relu"),
                     Dropout(0.1),
                     Dense(units=10, activation="relu"),
                     Dropout(0.1),
                     Dense(units=1),
                     Lambda(clip_steering)]


DONKEYNET_LAYERS = [DonkeyNetV0Layers, DonkeyNetV1Layers]


def get_donkey_net(version):
    return DONKEYNET_LAYERS[version]
