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
        self.threshold_prob = config.threshold_confidence
        self.replay_buffer = {"images": [], "sensor_data": [], "throttle_prob": []}
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
            inputs["sensor_input"] = self.store("sensor_data", sensor_data)
        action = self.predict(inputs)
        if len(action) > 1:
            steering, throttle_probs = action
            steering = np.squeeze(steering).item()
            throttle_probs = np.squeeze(throttle_probs)
            throttle_probs = self.store(key="throttle_prob", value=throttle_probs)
            throttle_probs = np.mean(np.squeeze(throttle_probs), axis=0)
            stop_prob = round(np.mean(throttle_probs[1]), 4)
            throttle_scale = max(0, (self.threshold_prob - stop_prob)/self.threshold_prob)
            throttle_dir = np.argmax(throttle_probs)-1
            if throttle_dir == 0 and stop_prob < self.threshold_prob:
                throttle_probs[1] = 0.0                            # Set no-throttle prob to zero
                throttle_dir = np.argmax(throttle_probs) - 1       # Select second best action
            throttle = throttle_scale*self.throttle*throttle_dir
        else:
            steering = action.item()
            throttle = self.throttle
            stop_prob = None
        return steering, throttle, stop_prob


class DonkeyNetClassifierController(BasePart):
    def __init__(self, graph, sess, model, config, buffer_size=10, sensor_only=False):
        self.graph = graph
        self.sess = sess
        self.model = model
        self.buffer_size = buffer_size
        self.replay_buffer = []
        self.sensor_only = sensor_only
        self.using_sensors = config.using_sensors
        self.threshold_confidence = config.threshold_confidence

    def store(self, value):
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(value)
        _buffer = np.array(self.replay_buffer)
        return _buffer

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
        avg_prob = np.mean(self.store(parked_prob))
        parked = (avg_prob >= self.threshold_confidence)
        throttle_scale = max(0.0, (self.threshold_confidence-avg_prob)/self.threshold_confidence)
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
