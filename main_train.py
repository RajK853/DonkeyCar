import os
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf_v1
from tensorflow.python.util import deprecation
from sklearn.model_selection import train_test_split

from src.model import DonkeyNet
from src.utils import load_data, blur_img, flip_img, data_generator, parse_args


tf_v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
deprecation._PRINT_DEPRECATION_WARNINGS = False


if __name__ == "__main__":
    print("######################################## DonkeyNet Model - Train ########################################")
    args = parse_args(mode="train")
    X, Y = [], []
    for data_dir in args.data_dirs:
        _X, _Y = load_data(data_dir, img_key=args.img_key, steering_key=args.steering_key,
                           throttle_key=args.throttle_key, force_process=args.force_process)
        _Y = np.reshape(_Y[:, 0], (_Y.shape[0], 1))                       # Remove throttle readings from actions
        X.extend(_X)
        Y.extend(_Y)
    random_seed = randint(0, 1000)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=random_seed, shuffle=True)
    # Model parameters
    input_shape = X_train[0].shape
    loss_func = tf_v1.losses.mean_squared_error
    optimizer = tf_v1.train.AdamOptimizer(learning_rate=args.lr)
    model = DonkeyNet(version=args.version, input_shape=input_shape, loss_func=loss_func, optimizer=optimizer)
    # Tensorflow session configuration
    GPU_OPTIONS = tf_v1.GPUOptions(per_process_gpu_memory_fraction=0.75)
    TF_CONFIG = tf_v1.ConfigProto(gpu_options=GPU_OPTIONS, allow_soft_placement=True)
    with tf_v1.Session(config=TF_CONFIG) as sess:
        sess.run(tf_v1.global_variables_initializer())
        if args.retrain_model:
            model.restore_model(sess, file_path=os.path.join(args.retrain_model, "model.chkpt"))
        # Add preprocessors
        preprocessors = []
        if args.add_flip:
            print("- Added flip preprocessor!")
            preprocessors.append(flip_img)
        if args.add_blur:
            print("- Added blur preprocessor!")
            preprocessors.append(blur_img)
        train_data_gen = data_generator(X_train, Y_train, epochs=args.epochs, batch_size=256,
                                        preprocessors=preprocessors)
        test_data_gen = data_generator(X_test, Y_test, batch_size=1)
        print("# Training the model!")
        train_losses = model.run(sess, data_gen=train_data_gen, training=True)
        mean_train_loss = np.mean(train_losses)
        print(f"  Mean training loss: {mean_train_loss:.3f}")
        print("# Testing the model")
        test_losses = model.run(sess, data_gen=test_data_gen, training=False)
        model.save_model(sess, os.path.join(args.save_model_path, "model.chkpt"))
        mean_test_loss = np.mean(test_losses)
        print(f"  Mean testing loss: {mean_test_loss:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=[20, 5], squeeze=False)
    axes[0, 0].plot(train_losses)
    axes[0, 0].set_title("Training losses")
    axes[0, 0].set_xlabel(f"Mean loss: {mean_train_loss:.3f}")
    axes[0, 1].hist(test_losses)
    axes[0, 1].set_title("Testing losses (histogram)")
    axes[0, 1].set_xlabel(f"Mean loss: {mean_test_loss:.3f}")
    plt_save_path = os.path.join(args.save_model_path, "Losses.jpg")
    plt.savefig(plt_save_path)
    print(f"- Graphs saved to {plt_save_path}")
    plt.show()
