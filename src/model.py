import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from .layers import get_donkey_net
from .utils import timer_wrapper


class Model:
    def __init__(self, scope, *, input_ph, layers, target_ph, loss_func, optimizer):
        self.scope = scope
        self.input_ph = input_ph
        self.network_output = self.build_network(layers)
        self.target_ph = target_ph
        self._saver = None
        self._trainable_vars = None
        self.loss = loss_func(self.target_ph, self.network_output)
        self.train_op = optimizer.minimize(self.loss, var_list=self.trainable_vars)

    def build_network(self, layers):
        output = self.input_ph
        with tf_v1.variable_scope(self.scope, reuse=True):
            for layer in layers:
                output = layer(output)
        print(f"Model {self.scope} with {len(layers)} layers built!\n")
        return output

    @property
    def input(self):
        return self.input_ph

    @property
    def output(self):
        return tf.squeeze(self.network_output)

    def predict(self, sess, inputs):
        outputs = sess.run(self.output, feed_dict={self.input_ph: inputs})
        return outputs

    @property
    def trainable_vars(self):
        if self._trainable_vars is None:
            self._trainable_vars = tf_v1.get_collection(tf_v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
            self._trainable_vars = sorted(self.trainable_vars, key=lambda var: var.name)
        return self._trainable_vars

    def run_batch(self, sess, batch_inputs, batch_targets, training=False):
        sess_runs = [self.loss, self.train_op] if training else [self.loss]
        loss, *_ = sess.run(sess_runs, feed_dict={self.input_ph: batch_inputs, self.target_ph: batch_targets})
        return loss

    @timer_wrapper
    def run(self, sess, data_gen, training=False):
        losses = [self.run_batch(sess, inputs, targets, training=training) for inputs, targets in data_gen]
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
        print(f"Model saved as '{file_path}'!")

    def restore_model(self, sess, file_path):
        """
        Restores varibles to the session
        args:
            file_path (str) : Source model file path to store variables (as checkpoint)
        """
        self.saver.restore(sess, file_path)
        print(f"Model restored from '{file_path}'!")


class DonkeyNet(Model):
    def __init__(self, version, *, input_shape, loss_func=None, optimizer=None):
        kwargs = {"input_ph": tf_v1.placeholder(tf.float32, shape=[None, *input_shape], name="images_ph"),
                  "layers": get_donkey_net(version),
                  "target_ph": tf_v1.placeholder(tf.float32, shape=[None, 1], name="target_actions_ph"),
                  "loss_func": loss_func or tf_v1.losses.mean_squared_error,
                  "optimizer": optimizer or tf_v1.train.AdamOptimizer(learning_rate=1e-4)}
        super(DonkeyNet, self).__init__(f"DonkeyNetV{version}Model", **kwargs)
