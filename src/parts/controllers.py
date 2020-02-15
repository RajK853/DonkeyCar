import os
import tensorflow as tf
from ..model import DonkeyNet
from ..utils import crop_img


class DonkeyNetController:
    def __init__(self, sess, input_shape, model_path, version=1, throttle=0.15):
        self.sess = sess
        self.throttle = throttle
        self.input_shape = input_shape
        self.model = DonkeyNet(version=version, input_shape=input_shape)
        # TODO: Initialize global variables somewhere
        sess.run(tf.global_variables_initializer())
        self.model.restore_model(sess, os.path.join(model_path, "model.chkpt"))

    def run(self, img_array):
        # Cropping layer added in tf instead
        # img_array = [crop_img(img_array, crop_dim=(slice(20, -1),))]
        steering = self.model.predict(self.sess, [img_array])
        return steering, self.throttle

    def shutdown(self):
        self.sess.close()


class RandomController:
    def __init__(self, env):
        self.env = env

    def run(self):
        steering, throttle = self.env.action_space.sample()
        return steering, throttle
