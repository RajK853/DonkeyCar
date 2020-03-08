import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
import numpy as np

from src import layers
from .utils import timer_wrapper

DONKEY_NET_PREFIX = "donkey_net_v"
DONKEY_NETS = [attr for attr in dir(layers) if attr.startswith(DONKEY_NET_PREFIX)]


def get_donkey_net(version):
    donkey_net_name = DONKEY_NETS[version]
    donkey_net_func = getattr(layers, donkey_net_name)
    return donkey_net_func


class Model:
    def __init__(self, scope, *, input_phs, network_func, target_phs, loss_func, optimizer):
        self.scope = scope
        self.input_phs = input_phs
        self.target_phs = target_phs
        self.network_outputs = self.build_network(network_func)
        self.input_num = len(input_phs)
        self.output_num = len(self.network_outputs)
        assert self.output_num == len(self.target_phs), \
            "Number of network output tensors and target_phs should be equal"
        self._saver = None
        self._trainable_vars = {}
        # TODO: Define loss function and training operation outside
        self.losses = []
        self.train_ops = []
        self.setup_loss(loss_func, optimizer)

    def setup_loss(self, loss_func, optimizer):
        for i in range(self.output_num):
            target_ph = self.target_phs[i]
            network_output = self.network_outputs[i]
            loss = loss_func(target_ph, network_output)
            var_scope = target_ph.name.split("_")[0]
            self.losses.append(loss)
            self.train_ops.append(optimizer.minimize(loss, var_list=self.trainable_vars(var_scope)))

    def build_network(self, network_func):
        print(f"# Building {network_func.__name__}")
        with tf_v1.variable_scope(self.scope, reuse=True):
            output = network_func(*self.input_phs)
        return output

    @property
    def input(self):
        return self.input_phs

    @property
    def output(self):
        return self.network_outputs

    def predict(self, sess, inputs):
        feed_dict = self.get_feed_dict(inputs)
        outputs = sess.run(self.output, feed_dict=feed_dict)
        outputs = np.squeeze(outputs)
        return outputs

    def trainable_vars(self, var_scope):
        if var_scope == "main":
            main_vars = []
        else:
            main_vars = self.trainable_vars("main")
        var_scope = "/".join((self.scope, var_scope))
        if var_scope not in self._trainable_vars.keys():
            trainable_vars = tf_v1.get_collection(tf_v1.GraphKeys.TRAINABLE_VARIABLES, scope=var_scope)
            trainable_vars.extend(main_vars)
            trainable_vars = sorted(trainable_vars, key=lambda var: var.name)
            self._trainable_vars[var_scope] = trainable_vars
            print(f"Loaded trainable variables in '{var_scope}'")
        return self._trainable_vars[var_scope]

    def get_feed_dict(self, inputs, targets=None):
        feed_dict = {self.input_phs[i]: inputs[i] for i in range(self.input_num)}
        if targets is not None:
            feed_dict.update({self.target_phs[i]: targets[:, i] for i in range(self.output_num)})
        return feed_dict

    def run_batch(self, sess, batch_inputs, batch_targets, training=False):
        sess_runs = self.losses + self.train_ops if training else self.losses
        feed_dict = self.get_feed_dict(batch_inputs, batch_targets)
        result = sess.run(sess_runs, feed_dict=feed_dict)
        loss = result[:self.output_num]
        return loss

    @timer_wrapper
    def run(self, sess, data_gen, training=False):
        losses = [self.run_batch(sess, inputs, targets, training=training) for inputs, targets in data_gen]
        losses = np.squeeze(losses)
        print()
        return losses

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf_v1.train.Saver(max_to_keep=10)
        return self._saver

    def save_model(self, sess, file_path):
        """
        Saves variables from the session
        args:
            file_path (str) : Destination model file path to store variables (as checkpoint)
        """
        self.saver.save(sess, file_path)
        print(f"# Model saved as '{file_path}'!")

    def restore_model(self, sess, file_path):
        """
        Restores variables to the session
        args:
            file_path (str) : Source model file path to store variables (as checkpoint)
        """
        self.saver.restore(sess, file_path)
        print(f"# Model restored from '{file_path}'!")


class DonkeyNet(Model):
    def __init__(self, version, *, input_shape, config, loss_func=None, optimizer=None):
        input_phs = [tf_v1.placeholder(tf.float32, shape=[None, *input_shape], name="images_ph")]
        if config.SEQUENCE_LENGTH > 1:
            sensor_shape = [None, config.SEQUENCE_LENGTH, config.SENSOR_NUM]
        else:
            sensor_shape = [None, config.SENSOR_NUM]
        target_shape = [None, 1]
        if config.INCLUDE_SENSORS:
            input_phs.append(tf_v1.placeholder(tf.float32, shape=sensor_shape, name="sensors_ph"))
        target_phs = [tf_v1.placeholder(tf.float32, shape=target_shape, name="steering_ph")]
        if config.INCLUDE_THROTTLE:
            target_phs.append(tf_v1.placeholder(tf.float32, shape=target_shape, name="throttle_ph"))
        loss_func = loss_func or tf_v1.losses.mean_squared_error
        optimizer = optimizer or tf_v1.train.AdamOptimizer(learning_rate=1e-4)
        network_func = get_donkey_net(version)
        super(DonkeyNet, self).__init__(f"DonkeyNetV{version}Model", input_phs=input_phs, target_phs=target_phs,
                                        network_func=network_func, loss_func=loss_func, optimizer=optimizer)
