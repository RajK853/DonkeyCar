import tensorflow.compat.v1 as tf_v1
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Cropping2D, Conv2D, Dense, MaxPool2D, Flatten, Dropout, Lambda, \
    Concatenate, LSTM, TimeDistributed as TD
from ..utils import clip_steering_tf, clip_throttle_tf


def donkey_model_v1(img_shape, sensor_shape, lr, **_):
    img_input = Input(shape=img_shape, dtype="float32", name="image_input")
    sensor_input = Input(shape=sensor_shape, dtype="float32", name="sensor_input")
    kernel_regularizer = l2(1e-3)
    default_kwargs = {"activation": "relu", "kernel_regularizer": kernel_regularizer}
    # Process image
    layer_out = Cropping2D(((0, 20), (0, 0)), data_format="channels_last")(img_input)
    layer_out = Conv2D(filters=24, kernel_size=5, strides=2, **default_kwargs)(layer_out)
    layer_out = Conv2D(filters=32, kernel_size=5, strides=2, **default_kwargs)(layer_out)
    layer_out = Conv2D(filters=64, kernel_size=5, strides=2, **default_kwargs)(layer_out)
    layer_out = Conv2D(filters=64, kernel_size=3, strides=2, **default_kwargs)(layer_out)
    layer_out = Flatten()(layer_out)
    layer_out = Dense(units=50, **default_kwargs)(layer_out)
    layer_out = Dense(units=25, **default_kwargs)(layer_out)
    layer_out = Dropout(0.2)(layer_out)
    # Process sensor data
    sensor_out = Dense(units=50, **default_kwargs)(sensor_input)
    sensor_out = Dense(units=50, **default_kwargs)(sensor_out)
    sensor_out = Dense(units=50, **default_kwargs)(sensor_out)
    sensor_out = Dense(units=25, **default_kwargs)(sensor_out)
    sensor_out = Dropout(0.2)(sensor_out)
    # Concatenate processed image and sensor data
    layer_out = Concatenate(axis=-1)([layer_out, sensor_out])
    # Compute steering
    steering = Dense(units=50, **default_kwargs)(layer_out)
    steering = Dense(units=50, **default_kwargs)(steering)
    steering = Dropout(0.2)(steering)
    steering = Dense(units=25, **default_kwargs)(steering)
    steering = Dense(units=1)(steering)
    steering = Lambda(clip_steering_tf, name="steering_output")(steering)
    model = Model(inputs=[img_input, sensor_input], outputs=[steering])
    # Loss function, optimizer and metrics
    loss = tf_v1.keras.losses.MeanSquaredError()
    optimizer = tf_v1.keras.optimizers.Adam(lr=lr)
    model.compile(loss=loss, optimizer=optimizer)
    return model


def donkey_model_v2(img_shape, sensor_shape, lr, **_):
    img_input = Input(shape=img_shape, dtype="float32", name="image_input")
    sensor_input = Input(shape=sensor_shape, dtype="float32", name="sensor_input")
    kernel_regularizer = l2(1e-3)
    default_kwargs = {"activation": "relu", "kernel_regularizer": kernel_regularizer}
    # Process image
    layer_out = TD(Cropping2D(((0, 20), (0, 0)), data_format="channels_last"))(img_input)
    layer_out = TD(Conv2D(filters=24, kernel_size=5, strides=2, **default_kwargs))(layer_out)
    layer_out = TD(Conv2D(filters=32, kernel_size=5, strides=2, **default_kwargs))(layer_out)
    layer_out = TD(Conv2D(filters=64, kernel_size=5, strides=2, **default_kwargs))(layer_out)
    layer_out = TD(Conv2D(filters=64, kernel_size=3, strides=2, **default_kwargs))(layer_out)
    layer_out = TD(Flatten())(layer_out)
    layer_out = LSTM(units=50, **default_kwargs)(layer_out)
    layer_out = Dense(units=25, **default_kwargs)(layer_out)
    layer_out = Dropout(0.2)(layer_out)
    # Process sensor data
    sensor_out = LSTM(units=50, **default_kwargs)(sensor_input)
    sensor_out = Dense(units=50, **default_kwargs)(sensor_out)
    sensor_out = Dense(units=50, **default_kwargs)(sensor_out)
    sensor_out = Dense(units=25, **default_kwargs)(sensor_out)
    sensor_out = Dropout(0.2)(sensor_out)
    # Concatenate processed image and sensor data
    layer_out = Concatenate(axis=-1)([layer_out, sensor_out])
    # Compute steering
    steering = Dense(units=50, **default_kwargs)(layer_out)
    steering = Dense(units=50, **default_kwargs)(steering)
    steering = Dropout(0.2)(steering)
    steering = Dense(units=25, **default_kwargs)(steering)
    steering = Dense(units=1)(steering)
    steering = Lambda(clip_steering_tf, name="steering_output")(steering)
    model = Model(inputs=[img_input, sensor_input], outputs=[steering])
    # Loss function, optimizer and metrics
    loss = tf_v1.keras.losses.MeanSquaredError()
    optimizer = tf_v1.keras.optimizers.Adam(lr=lr)
    model.compile(loss=loss, optimizer=optimizer)
    return model


def donkey_model_v3(img_shape, sensor_shape, lr, **_):
    img_input = Input(shape=img_shape, dtype="float32", name="image_input")
    sensor_input = Input(shape=sensor_shape, dtype="float32", name="sensor_input")
    kernel_regularizer = l2(1e-3)
    default_kwargs = {"activation": "relu", "kernel_regularizer": kernel_regularizer}
    # Process image
    layer_out = TD(Cropping2D(((0, 20), (0, 0)), data_format="channels_last"))(img_input)
    layer_out = TD(Conv2D(filters=24, kernel_size=5, strides=2, **default_kwargs))(layer_out)
    layer_out = TD(Conv2D(filters=32, kernel_size=5, strides=2, **default_kwargs))(layer_out)
    layer_out = TD(Conv2D(filters=64, kernel_size=5, strides=2, **default_kwargs))(layer_out)
    layer_out = TD(Conv2D(filters=64, kernel_size=3, strides=2, **default_kwargs))(layer_out)
    layer_out = TD(Flatten())(layer_out)
    layer_out = LSTM(units=50, **default_kwargs)(layer_out)
    layer_out = Dense(units=25, **default_kwargs)(layer_out)
    layer_out = Dropout(0.2)(layer_out)
    # Process sensor data
    sensor_out = LSTM(units=50, **default_kwargs)(sensor_input)
    sensor_out = Dense(units=50, **default_kwargs)(sensor_out)
    sensor_out = Dense(units=50, **default_kwargs)(sensor_out)
    sensor_out = Dense(units=25, **default_kwargs)(sensor_out)
    sensor_out = Dropout(0.2)(sensor_out)
    # Concatenate processed image and sensor data
    layer_out = Concatenate(axis=-1)([layer_out, sensor_out])
    # Compute steering
    steering = Dense(units=50, **default_kwargs)(layer_out)
    steering = Dense(units=50, **default_kwargs)(steering)
    steering = Dense(units=50, **default_kwargs)(steering)
    steering = Dropout(0.2)(steering)
    steering = Dense(units=25, **default_kwargs)(steering)
    steering = Dense(units=1)(steering)
    steering = Lambda(clip_steering_tf, name="steering_output")(steering)

    # Compute steering
    throttle = Dense(units=50, **default_kwargs)(layer_out)
    throttle = Dense(units=50, **default_kwargs)(throttle)
    throttle = Dropout(0.2)(throttle)
    throttle = Dense(units=25, **default_kwargs)(throttle)
    throttle = Dense(units=3, activation="softmax", name="throttle_output")(throttle)

    model = Model(inputs=[img_input, sensor_input], outputs=[steering, throttle])
    # Loss function, optimizer and metrics
    loss = {"steering_output": tf_v1.keras.losses.MeanSquaredError(),
            "throttle_output": tf_v1.keras.losses.CategoricalCrossentropy()}
    metrics = {"throttle_output": "accuracy"}
    optimizer = tf_v1.keras.optimizers.Adam(lr=lr)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def donkey_classifier_v1(sensor_shape, lr, **_):
    sensor_input = Input(shape=sensor_shape, dtype="float32", name="sensor_input")
    kernel_regularizer = l2(1e-3)
    default_kwargs = {"activation": "relu", "kernel_regularizer": kernel_regularizer}
    layer_out = Dense(units=128, **default_kwargs)(sensor_input)
    layer_out = Dense(units=128, **default_kwargs)(layer_out)
    layer_out = Dense(units=64, **default_kwargs)(layer_out)
    layer_out = Dropout(0.25)(layer_out)
    layer_out = Dense(units=64, **default_kwargs)(layer_out)
    layer_out = Dense(units=128, **default_kwargs)(layer_out)
    layer_out = Dense(units=128, **default_kwargs)(layer_out)
    parked_output = Dense(units=2, activation="softmax", name="parked_output")(layer_out)
    model = Model(inputs=[sensor_input], outputs=parked_output)
    # Loss function, optimizer and metrics
    loss = tf_v1.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf_v1.keras.optimizers.Adam(lr=lr)
    metrics = ["accuracy"]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def donkey_classifier_v2(img_shape, sensor_shape, lr, **_):
    img_input = Input(shape=img_shape, dtype="float32", name="image_input")
    sensor_input = Input(shape=sensor_shape, dtype="float32", name="sensor_input")
    kernel_regularizer = l2(1e-3)
    default_kwargs = {"activation": "relu", "kernel_regularizer": kernel_regularizer}
    # Process image
    layer_out = Cropping2D(((0, 20), (0, 0)), data_format="channels_last")(img_input)
    layer_out = Conv2D(filters=24, kernel_size=5, strides=2, **default_kwargs)(layer_out)
    layer_out = Conv2D(filters=32, kernel_size=5, strides=2, **default_kwargs)(layer_out)
    layer_out = Conv2D(filters=64, kernel_size=5, strides=2, **default_kwargs)(layer_out)
    layer_out = Conv2D(filters=64, kernel_size=3, strides=2, **default_kwargs)(layer_out)
    layer_out = Flatten()(layer_out)
    layer_out = Dense(units=50, **default_kwargs)(layer_out)
    layer_out = Dense(units=25, **default_kwargs)(layer_out)
    layer_out = Dropout(0.2)(layer_out)

    sensor_out = Dense(units=128, **default_kwargs)(sensor_input)
    sensor_out = Dense(units=128, **default_kwargs)(sensor_out)
    sensor_out = Dense(units=64, **default_kwargs)(sensor_out)
    sensor_out = Dropout(0.25)(sensor_out)

    # Concatenate processed image and sensor data
    layer_out = Concatenate(axis=-1)([layer_out, sensor_out])
    layer_out = Dense(units=64, **default_kwargs)(layer_out)
    layer_out = Dense(units=128, **default_kwargs)(layer_out)
    layer_out = Dense(units=128, **default_kwargs)(layer_out)
    parked_output = Dense(units=2, activation="softmax", name="parked_output")(layer_out)
    model = Model(inputs=[img_input, sensor_input], outputs=parked_output)
    # Loss function, optimizer and metrics
    loss = tf_v1.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf_v1.keras.optimizers.Adam(lr=lr)
    metrics = ["accuracy"]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
