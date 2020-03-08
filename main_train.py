import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf_v1
from tensorflow.python.util import deprecation

import config
from src.model import DonkeyNet
from src.utils import load_data, flip_img, data_generator, parse_args

# Tensorflow session configuration
tf_v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
deprecation._PRINT_DEPRECATION_WARNINGS = False
GPU_OPTIONS = tf_v1.GPUOptions(per_process_gpu_memory_fraction=0.75)
TF_CONFIG = tf_v1.ConfigProto(gpu_options=GPU_OPTIONS, allow_soft_placement=True)


if __name__ == "__main__":
    print("######################################## DonkeyNet Model - Train ########################################")
    args = parse_args(mode="train")
    train_images, train_actions, train_sensor_data = [], [], []
    test_images, test_actions, test_sensor_data = [], [], []
    for mode in ("train", "test"):
        if mode == "train":
            X, Y, Z = train_images, train_actions, train_sensor_data
            data_dirs = args.train_data_dirs
        else:
            X, Y, Z = test_images, test_actions, test_sensor_data
            data_dirs = args.test_data_dirs
        for data_dir in data_dirs:
            images, sensor_data, actions = load_data(data_dir, config, force_process=args.force_process)
            X.extend(images)
            Y.extend(actions)
            Z.extend(sensor_data)
    # Model parameters
    input_shape = train_images[0].shape
    if config.SEQUENCE_LENGTH > 1:
        input_shape = (config.SEQUENCE_LENGTH,) + input_shape
    loss_func = tf_v1.losses.mean_squared_error
    optimizer = tf_v1.train.AdamOptimizer(learning_rate=args.lr)
    model = DonkeyNet(version=args.version, input_shape=input_shape, config=config, loss_func=loss_func,
                      optimizer=optimizer)
    with tf_v1.Session(config=TF_CONFIG) as sess:
        sess.run(tf_v1.global_variables_initializer())
        if args.retrain_model:
            model.restore_model(sess, file_path=os.path.join(args.retrain_model, "model.chkpt"))
        # Add preprocessors
        preprocessors = []
        if args.add_flip:
            print("- Added flip preprocessor!")
            preprocessors.append(flip_img)
        train_data_gen = data_generator(train_images, train_actions, sensor_data=train_sensor_data, epochs=args.epochs,
                                        batch_size=256, preprocessors=preprocessors, sequence=config.SEQUENCE_LENGTH)
        test_data_gen = data_generator(test_images, test_actions, sensor_data=test_sensor_data, batch_size=1,
                                       sequence=config.SEQUENCE_LENGTH)
        print("# Training the model!")
        train_losses = model.run(sess, data_gen=train_data_gen, training=True)
        mean_train_loss = np.mean(train_losses)
        print(f"  Mean training loss: {mean_train_loss:.3f}")
        print("# Testing the model")
        test_losses = model.run(sess, data_gen=test_data_gen, training=False)
        model.save_model(sess, os.path.join(args.save_model_path, "model.chkpt"))
        mean_test_loss = np.mean(test_losses)
        print(f"  Mean testing loss: {mean_test_loss:.3f}")

    legends = ["steering_loss"] if not config.INCLUDE_THROTTLE else ["steering_loss", "throttle_loss"]
    fig, axes = plt.subplots(1, 2, figsize=[20, 5], squeeze=False)
    axes[0, 0].plot(train_losses)
    axes[0, 0].set_title("Training losses")
    axes[0, 0].set_xlabel(f"Mean loss: {mean_train_loss:.3f}")
    axes[0, 0].legend(legends)
    axes[0, 1].hist(test_losses)
    axes[0, 1].set_title("Testing losses (histogram)")
    axes[0, 1].set_xlabel(f"Mean loss: {mean_test_loss:.3f}")
    axes[0, 1].legend(legends)
    plt_save_path = os.path.join(args.save_model_path, "Losses.jpg")
    plt.savefig(plt_save_path)
    print(f"- Graphs saved to {plt_save_path}")
    plt.show()
