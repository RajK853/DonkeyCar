import numpy as np

from .base import BasePart
from ..image_processing import normalize_images


class DonkeyNetController(BasePart):
    def __init__(self, graph, sess, model, config, throttle=0.15):
        self.graph = graph
        self.sess = sess
        self.throttle = throttle
        self.sequence_len = config.sequence_length
        self.using_sensors = config.using_sensors
        self.replay_buffer = {"images": [], "sensor_data": []}
        self.model = model

    def store(self, key, value):
        if len(self.replay_buffer[key]) >= self.sequence_len:
            self.replay_buffer[key].pop(0)
        while len(self.replay_buffer[key]) < self.sequence_len:
            self.replay_buffer[key].append(value)
        _buffer = np.array(self.replay_buffer[key])
        if self.sequence_len > 1:
            _buffer = _buffer.reshape((1, *_buffer.shape))
        return _buffer

    def predict(self, inputs):
        with self.graph.as_default():
            with self.sess.as_default():
                output = self.model.predict(inputs)
                return output

    def run(self, img_array, *sensor_data):
        inputs = {}
        norm_img_array = normalize_images(img_array)
        img_data = self.store("images", norm_img_array)
        inputs["image_input"] = img_data
        if self.using_sensors:
            if any([data is None for data in sensor_data]):
                sensor_data = [0.0 for _ in range(len(sensor_data))]
            sensor_data = self.store("sensor_data", sensor_data)
            inputs["sensor_input"] = sensor_data
        action = self.predict(inputs)
        if len(action) > 1:
            steering, throttle = np.squeeze(action)
        else:
            steering = action.item()
            throttle = self.throttle
        return steering, throttle


class DonkeyNetClassifierController(BasePart):
    def __init__(self, graph, sess, model, config, sensor_only=False):
        self.graph = graph
        self.sess = sess
        self.model = model
        self.sensor_only = sensor_only
        self.using_sensors = config.using_sensors
        self.threshold_confidence = config.threshold_confidence

    def predict(self, inputs):
        with self.graph.as_default():
            with self.sess.as_default():
                output = self.model.predict(inputs)
                output = np.squeeze(output)
                return output

    def run(self, img_array, *sensor_data):
        if img_array is None:
            return 0.0, False, 1.0
        inputs = {}
        if not self.sensor_only:
            img_data = normalize_images(img_array)
            inputs["image_input"] = img_data.reshape(1, *img_data.shape)
        if self.using_sensors:
            if any([data is None for data in sensor_data]):
                sensor_data = [0.0 for _ in range(len(sensor_data))]
            sensor_data = np.array(sensor_data)
            sensor_data = sensor_data.reshape(1, *sensor_data.shape)
            inputs["sensor_input"] = sensor_data
        parked_prob = self.predict(inputs)[1]
        parked = (parked_prob >= self.threshold_confidence)
        throttle_scale = max(0.0, (self.threshold_confidence-parked_prob)/self.threshold_confidence)
        return round(parked_prob, 4), parked, throttle_scale


class NullController(BasePart):
    def __init__(self):
        pass

    @staticmethod
    def run():
        steering = throttle = 0.0
        return steering, throttle


class RandomController(BasePart):
    def __init__(self, env):
        self.env = env

    def run(self):
        steering, throttle = self.env.action_space.sample()
        return steering, throttle

    def shutdown(self):
        self.env.close()
