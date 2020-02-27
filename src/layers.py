from tensorflow.compat.v1.image import resize as tf_img_resize
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Cropping2D, Conv2D, Dense, MaxPool2D, Flatten, Dropout, Lambda, ConvLSTM2D
from .utils import clip_steering_tf


def donkey_net_v0(input_ph):
    layer_out = Cropping2D(((30, 0), (0, 0)), data_format="channels_last")(input_ph)
    layer_out = Conv2D(filters=64, kernel_size=7, strides=2, activation="relu", kernel_regularizer=l2(0.001))(layer_out)
    layer_out = MaxPool2D(pool_size=(4, 4), strides=2)(layer_out)
    layer_out = Conv2D(filters=32, kernel_size=5, strides=2, activation="relu", kernel_regularizer=l2(0.001))(layer_out)
    layer_out = MaxPool2D(pool_size=(2, 2), strides=2)(layer_out)
    layer_out = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", kernel_regularizer=l2(0.001))(layer_out)
    layer_out = Flatten()(layer_out)
    layer_out = Dense(units=64, activation="relu", kernel_regularizer=l2(0.001))(layer_out)
    layer_out = Dropout(0.75)(layer_out)
    layer_out = Dense(units=32, activation="relu", kernel_regularizer=l2(0.001))(layer_out)
    layer_out = Dense(units=1)(layer_out)
    layer_out = Lambda(clip_steering_tf)(layer_out)
    return layer_out


def donkey_net_v1(input_ph):
    layer_out = Cropping2D(((30, 0), (0, 0)), data_format="channels_last")(input_ph)
    layer_out = Conv2D(filters=24, kernel_size=5, strides=2, activation="relu")(layer_out)
    layer_out = Conv2D(filters=32, kernel_size=5, strides=2, activation="relu")(layer_out)
    layer_out = Conv2D(filters=64, kernel_size=5, strides=2, activation="relu")(layer_out)
    layer_out = Conv2D(filters=64, kernel_size=3, strides=1, activation="relu")(layer_out)
    layer_out = Conv2D(filters=64, kernel_size=3, strides=1, activation="relu")(layer_out)
    layer_out = Flatten()(layer_out)
    layer_out = Dense(units=100, activation="relu")(layer_out)
    layer_out = Dropout(0.1)(layer_out)
    layer_out = Dense(units=50, activation="relu")(layer_out)
    layer_out = Dropout(0.1)(layer_out)
    layer_out = Dense(units=10, activation="relu")(layer_out)
    layer_out = Dropout(0.1)(layer_out)
    layer_out = Dense(units=1)(layer_out)
    layer_out = Lambda(clip_steering_tf)(layer_out)
    return layer_out


def donkey_net_v2(input_ph):
    layer_out = Cropping2D(((30, 0), (0, 0)), data_format="channels_last")(input_ph)
    layer_out = tf_img_resize(layer_out, size=(50, 80), preserve_aspect_ratio=True)
    layer_out = Conv2D(filters=64, kernel_size=5, strides=2, activation="relu")(layer_out)
    layer_out = Conv2D(filters=32, kernel_size=5, strides=2, activation="relu")(layer_out)
    layer_out = Conv2D(filters=8, kernel_size=3, strides=2, activation="relu")(layer_out)
    layer_out = Conv2D(filters=8, kernel_size=3, strides=1, activation="relu")(layer_out)
    layer_out = Flatten()(layer_out)
    layer_out = Dense(units=64, activation="relu")(layer_out)
    layer_out = Dropout(0.1)(layer_out)
    layer_out = Dense(units=16, activation="relu")(layer_out)
    layer_out = Dropout(0.1)(layer_out)
    layer_out = Dense(units=1)(layer_out)
    layer_out = Lambda(clip_steering_tf)(layer_out)
    return layer_out
