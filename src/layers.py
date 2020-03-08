import tensorflow.compat.v1 as tf_v1
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Cropping2D, Conv2D, Dense, MaxPool2D, Flatten, Dropout, Lambda, Concatenate, \
    LSTM, TimeDistributed as TD
from .utils import clip_steering_tf, clip_throttle_tf


def donkey_net_v0(input_ph, *_):
    with tf_v1.variable_scope("main", reuse=True):
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

    with tf_v1.variable_scope("steering", reuse=True):
        steering = Dense(units=1, activation="linear")(layer_out)
        steering = Lambda(clip_steering_tf)(steering)
    return [steering]


def donkey_net_v1(input_ph, *_):
    with tf_v1.variable_scope("main", reuse=True):
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

    with tf_v1.variable_scope("steering", reuse=True):
        steering = Dense(units=1, activation="linear")(layer_out)
        steering = Lambda(clip_steering_tf)(steering)
    return [steering]


def donkey_net_v2(input_ph, *_):
    with tf_v1.variable_scope("main", reuse=True):
        layer_out = Cropping2D(((30, 0), (0, 0)), data_format="channels_last")(input_ph)
        layer_out = tf_v1.image.resize(layer_out, size=(50, 80), preserve_aspect_ratio=True)
        layer_out = Conv2D(filters=64, kernel_size=5, strides=2, activation="relu")(layer_out)
        layer_out = Conv2D(filters=32, kernel_size=5, strides=2, activation="relu")(layer_out)
        layer_out = Conv2D(filters=8, kernel_size=3, strides=2, activation="relu")(layer_out)
        layer_out = Conv2D(filters=8, kernel_size=3, strides=1, activation="relu")(layer_out)
        layer_out = Flatten()(layer_out)
        layer_out = Dense(units=64, activation="relu")(layer_out)
        layer_out = Dropout(0.1)(layer_out)
        layer_out = Dense(units=16, activation="relu")(layer_out)
        layer_out = Dropout(0.1)(layer_out)

    with tf_v1.variable_scope("steering", reuse=True):
        steering = Dense(units=1, activation="linear")(layer_out)
        steering = Lambda(clip_steering_tf)(steering)
    return [steering]


def donkey_net_v3(input_ph, *_):
    with tf_v1.variable_scope("main", reuse=True):
        layer_out = Cropping2D(((50, 0), (0, 0)), data_format="channels_last")(input_ph)
        layer_out = Conv2D(filters=64, kernel_size=5, strides=2, activation="relu")(layer_out)
        layer_out = Conv2D(filters=32, kernel_size=5, strides=2, activation="relu")(layer_out)
        layer_out = Conv2D(filters=8, kernel_size=3, strides=2, activation="relu")(layer_out)
        layer_out = Conv2D(filters=8, kernel_size=3, strides=1, activation="relu")(layer_out)
        layer_out = Flatten()(layer_out)
        layer_out = Dense(units=64, activation="relu")(layer_out)
        layer_out = Dropout(0.1)(layer_out)
        layer_out = Dense(units=16, activation="relu")(layer_out)
        layer_out = Dropout(0.1)(layer_out)

    with tf_v1.variable_scope("steering", reuse=True):
        steering = Dense(units=1, activation="linear")(layer_out)
        steering = Lambda(clip_steering_tf)(steering)
    return [steering]


def donkey_net_v4(input_ph, *_):
    with tf_v1.variable_scope("main", reuse=True):
        layer_out = Cropping2D(((40, 0), (0, 0)), data_format="channels_last")(input_ph)
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

    with tf_v1.variable_scope("steering", reuse=True):
        steering = Dense(units=1, activation="linear")(layer_out)
        steering = Lambda(clip_steering_tf)(steering)
    return [steering]


def donkey_net_v5(input_ph, *_):
    with tf_v1.variable_scope("main", reuse=True):
        layer_out = Cropping2D(((40, 0), (0, 0)), data_format="channels_last")(input_ph)
        layer_out = Conv2D(filters=24, kernel_size=5, strides=2, activation="relu")(layer_out)
        layer_out = Conv2D(filters=32, kernel_size=5, strides=2, activation="relu")(layer_out)
        layer_out = Conv2D(filters=64, kernel_size=5, strides=2, activation="relu")(layer_out)
        layer_out = Conv2D(filters=64, kernel_size=3, strides=1, activation="relu")(layer_out)
        layer_out = Conv2D(filters=64, kernel_size=3, strides=1, activation="relu")(layer_out)
        layer_out = Flatten()(layer_out)
        layer_out = Dense(units=100, activation="relu")(layer_out)
        layer_out = Dropout(0.1)(layer_out)

    with tf_v1.variable_scope("steering", reuse=True):
        steering = Dense(units=50, activation="relu")(layer_out)
        steering = Dropout(0.1)(steering)
        steering = Dense(units=1, activation="linear")(steering)
        steering = Lambda(clip_steering_tf)(steering)

    with tf_v1.variable_scope("throttle", reuse=True):
        throttle = Dense(units=50, activation="relu")(layer_out)
        throttle = Dropout(0.1)(throttle)
        throttle = Dense(units=1, activation="linear")(throttle)
        throttle = Lambda(clip_throttle_tf)(throttle)
    return [steering, throttle]


def donkey_net_v6(input_ph, sensor_ph, *_):
    with tf_v1.variable_scope("main", reuse=True):
        layer_out = Cropping2D(((40, 0), (0, 0)), data_format="channels_last")(input_ph)
        layer_out = Conv2D(filters=24, kernel_size=5, strides=2, activation="relu")(layer_out)
        layer_out = Conv2D(filters=32, kernel_size=5, strides=2, activation="relu")(layer_out)
        layer_out = Conv2D(filters=64, kernel_size=5, strides=2, activation="relu")(layer_out)
        layer_out = Conv2D(filters=64, kernel_size=3, strides=1, activation="relu")(layer_out)
        layer_out = Conv2D(filters=64, kernel_size=3, strides=1, activation="relu")(layer_out)
        layer_out = Flatten()(layer_out)
        layer_out = Dense(units=100, activation="relu")(layer_out)
        layer_out = Dropout(0.1)(layer_out)
        layer_out = Concatenate(axis=1)([layer_out, sensor_ph])

    with tf_v1.variable_scope("steering", reuse=True):
        steering = Dense(units=50, activation="relu")(layer_out)
        steering = Dropout(0.1)(steering)
        steering = Dense(units=1, activation="linear")(steering)
        steering = Lambda(clip_steering_tf)(steering)

    with tf_v1.variable_scope("throttle", reuse=True):
        throttle = Dense(units=50, activation="relu")(layer_out)
        throttle = Dropout(0.1)(throttle)
        throttle = Dense(units=1, activation="linear")(throttle)
        throttle = Lambda(clip_throttle_tf)(throttle)
    return [steering, throttle]


def donkey_net_v7(input_ph, sensor_ph, *_):
    with tf_v1.variable_scope("main", reuse=True):
        layer_out = TD(Cropping2D(((40, 0), (0, 0)), data_format="channels_last"))(input_ph)
        layer_out = TD(Conv2D(filters=24, kernel_size=5, strides=2, activation="relu"))(layer_out)
        layer_out = TD(Conv2D(filters=32, kernel_size=5, strides=2, activation="relu"))(layer_out)
        layer_out = TD(Conv2D(filters=64, kernel_size=5, strides=2, activation="relu"))(layer_out)
        layer_out = TD(Conv2D(filters=64, kernel_size=3, strides=1, activation="relu"))(layer_out)
        layer_out = TD(Conv2D(filters=64, kernel_size=3, strides=1, activation="relu"))(layer_out)
        layer_out = TD(Flatten())(layer_out)
        layer_out = TD(Dense(units=50, activation="relu"))(layer_out)
        layer_out = TD(Dropout(0.1))(layer_out)

        sensor_out = TD(Dense(50))(sensor_ph)
        layer_out = Concatenate(axis=1)([layer_out, sensor_out])
        layer_out = LSTM(32, return_sequences=True)(layer_out)
        layer_out = Dropout(0.1)(layer_out)
        layer_out = LSTM(32, return_sequences=False)(layer_out)

    with tf_v1.variable_scope("steering", reuse=True):
        steering = Dense(units=25, activation="relu")(layer_out)
        steering = Dropout(0.1)(steering)
        steering = Dense(units=1, activation="linear")(steering)
        steering = Lambda(clip_steering_tf)(steering)

    with tf_v1.variable_scope("throttle", reuse=True):
        throttle = Dense(units=50, activation="relu")(layer_out)
        throttle = Dropout(0.1)(throttle)
        throttle = Dense(units=50, activation="relu")(throttle)
        throttle = Dropout(0.1)(throttle)
        throttle = Dense(units=1, activation="linear")(throttle)
        throttle = Lambda(clip_throttle_tf)(throttle)
    return [steering, throttle]


def donkey_net_v8(input_ph, *_):
    with tf_v1.variable_scope("main", reuse=True):
        layer_out = TD(Cropping2D(((40, 0), (0, 0)), data_format="channels_last"))(input_ph)
        layer_out = TD(Conv2D(filters=24, kernel_size=5, strides=2, activation="relu"))(layer_out)
        layer_out = TD(Conv2D(filters=32, kernel_size=5, strides=2, activation="relu"))(layer_out)
        layer_out = TD(Conv2D(filters=64, kernel_size=5, strides=2, activation="relu"))(layer_out)
        layer_out = TD(Conv2D(filters=64, kernel_size=3, strides=1, activation="relu"))(layer_out)
        layer_out = TD(Conv2D(filters=64, kernel_size=3, strides=1, activation="relu"))(layer_out)
        layer_out = TD(Flatten())(layer_out)
        layer_out = TD(Dense(units=50, activation="relu"))(layer_out)
        layer_out = TD(Dropout(0.1))(layer_out)
        layer_out = LSTM(32, return_sequences=True)(layer_out)
        layer_out = Dropout(0.1)(layer_out)
        layer_out = LSTM(32, return_sequences=False)(layer_out)

    with tf_v1.variable_scope("steering", reuse=True):
        steering = Dense(units=50, activation="relu")(layer_out)
        steering = Dropout(0.1)(steering)
        steering = Dense(units=1, activation="linear")(steering)
        steering = Lambda(clip_steering_tf)(steering)

    with tf_v1.variable_scope("throttle", reuse=True):
        throttle = Dense(units=50, activation="relu")(layer_out)
        throttle = Dropout(0.1)(throttle)
        throttle = Dense(units=1, activation="linear")(throttle)
        throttle = Lambda(clip_throttle_tf)(throttle)
    return [steering, throttle]
