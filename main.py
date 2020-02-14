import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf_v1
from sklearn.model_selection import train_test_split

from src.model import DonkeyNet
from src.utils import load_data, blur_img, flip_img, data_generator, parse_args

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_model_path, exist_ok=True)
    # Load and process data
    X, Y = load_data(args.data_dir, img_key=args.img_key, steering_key=args.steering_key, throttle_key=args.throttle_key,
                     extensions=[".json"], exclude_files=["meta.json"], crop_dim=(slice(20, -1),),
                     verbose=args.verbose)
    Y = np.reshape(Y[:, 0], (len(Y), 1))                  # Remove throttle readings from actions
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=853)
    # Model parameters
    input_shape = X_train[0].shape
    loss_func = tf_v1.losses.mean_squared_error
    optimizer = tf_v1.train.AdamOptimizer(learning_rate=1e-4)
    model = DonkeyNet(version=args.version, input_shape=input_shape, loss_func=loss_func, optimizer=optimizer)
    # Tensorflow session configuration
    GPU_OPTIONS = tf_v1.GPUOptions(per_process_gpu_memory_fraction=0.75)
    TF_CONFIG = tf_v1.ConfigProto(gpu_options=GPU_OPTIONS, allow_soft_placement=True)
    with tf_v1.Session(config=TF_CONFIG) as sess:
        sess.run(tf_v1.global_variables_initializer())
        if args.retrain_model:
            model.restore_model(sess, file_path=os.path.join(args.retrain_model, "model.chkpt"))
        train_data_gen = data_generator(X_train, Y_train, epochs=args.epochs, batch_size=256,
                                        preprocessors=[blur_img])
        test_data_gen = data_generator(X_test, Y_test, batch_size=1)
        print("Training the model!")
        train_losses = model.run(sess, data_gen=train_data_gen, training=True)
        print(f"Mean training loss: {np.mean(train_losses):.3f}")
        print("Testing the model")
        test_losses = model.run(sess, data_gen=test_data_gen, training=False)
        model.save_model(sess, os.path.join(args.save_model_path, "model.chkpt"))
        print(f"Mean testing loss: {np.mean(test_losses):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=[20, 5], squeeze=False)
    axes[0, 0].set_title("Training losses")
    axes[0, 0].plot(train_losses)
    axes[0, 1].set_title("Testing losses")
    axes[0, 1].plot(test_losses)
    plt.show()
