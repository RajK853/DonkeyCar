import os
from numpy import array as np_array
import tensorflow.compat.v1 as tf_v1
from ..model import DonkeyNet
from ..utils import normalize_images


class DonkeyNetController:
    def __init__(self, sess, input_shape, model_path, config, version=1, throttle=0.15):
        self.sess = sess
        self.throttle = throttle
        self.version = version
        self.input_shape = input_shape
        self.buffer_size = config.SEQUENCE_LENGTH
        self.replay_buffer = {"images":[], "sensor_data":[]}
        self.model = DonkeyNet(version=version, input_shape=input_shape, config=config)
        self.output_size = self.model.output_num
        sess.run(tf_v1.global_variables_initializer())
        self.model.restore_model(sess, os.path.join(model_path, "model.chkpt"))

    def store(self, key, value):
        buffer = self.replay_buffer[key]
        if len(buffer) > self.buffer_size:
            buffer.pop(0)
        while len(buffer) < self.buffer_size:
            buffer.append(value)
        return np_array(buffer)

    def run(self, img_array, *sensor_data):
        norm_img_array = normalize_images(img_array)
        if self.buffer_size > 1:
            img_data = self.store("images", norm_img_array)
        else:
            img_data = norm_img_array
        if len(sensor_data) > 0:
            if any([data is None for data in sensor_data]):
                sensor_data = [0.0 for _ in range(len(sensor_data))]
            if self.buffer_size > 1:
                sensor_data = self.store("sensor_data", sensor_data)
            else:
                sensor_data = np_array(sensor_data)
            inputs = [[img_data], [sensor_data]]
        else:
            inputs = [[img_data]]
        action = self.model.predict(self.sess, inputs)
        if self.output_size == 1:
            steering = action
            throttle = self.throttle
        else:
            steering, throttle = action
        return steering, throttle

    def shutdown(self):
        self.sess.close()


class RandomController:
    def __init__(self, env):
        self.env = env

    def run(self):
        steering, throttle = self.env.action_space.sample()
        return steering, throttle
