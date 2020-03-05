import os
import tensorflow as tf
from ..model import DonkeyNet
from ..utils import normalize_images


class DonkeyNetController:
    def __init__(self, sess, input_shape, model_path, version=1, throttle=0.15):
        self.sess = sess
        self.throttle = throttle
        self.version = version
        self.input_shape = input_shape
        self.model = DonkeyNet(version=version, input_shape=input_shape)
        self.output_size = self.model.meta["output_size"]
        sess.run(tf.global_variables_initializer())
        self.model.restore_model(sess, os.path.join(model_path, "model.chkpt"))

    def run(self, img_array):
        norm_img_array = normalize_images(img_array)
        action = self.model.predict(self.sess, [norm_img_array])
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
