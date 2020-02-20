import os
import cv2
import argparse
import numpy as np
import ujson as json
from time import time
from numba import jit, vectorize
from functools import wraps
from sklearn.utils import shuffle
from .progress_bar import ProgressBar

STEER_LIMIT_LEFT = -1.0
STEER_LIMIT_RIGHT = 1.0
THROTTLE_MIN = 0.0
THROTTLE_MAX = 5.0


def timer_wrapper(func):
    @wraps(func)
    def wrapper_func(*args, **kwargs):
        t0 = time()
        result = func(*args, **kwargs)
        dt = time()-t0
        print(f"- Function '{func.__name__}' took {dt:.3f} seconds!")
        return result
    return wrapper_func


def crop_img(img, crop_dim):
    assert all(isinstance(crop, slice) for crop in crop_dim), f"Crop_dim should be an slice, not {type(crop_dim)}"
    img = img[crop_dim]
    return img


def flip_img(img, action):
    flipped_img = cv2.flip(img, 1)
    # Make changes in the copy of action array
    action = action.copy()
    action[0] *= -1              # Flip steering direction
    # TODO: Swap right and left ultrasonic sensor readings
    return flipped_img, action


def blur_img(img, action):
    img = cv2.GaussianBlur(img, (9, 9), 0)
    return img, action


def normalize_images(images):
    return images/255.0


def denormalize_images(images):
    images = images * 255.0
    return images.astype(np.uint8)


def load_img(img_file, cvt_color=None):
    img = cv2.imread(img_file)
    if cvt_color:
        img = cv2.cvtColor(img, cvt_color)
    return img


def load_json(json_file, include_keys=None):
    with open(json_file, "r") as json_f:
        json_data = json.load(json_f)
    if include_keys:
        json_data = {key: json_data[key] for key in include_keys}
    return json_data


def matches_any_extension(file_name, extensions):
    for suffix in extensions:
        if file_name.endswith(suffix):
            return True
    return False


def list_dir(dir_path, extensions=None, exclude_files=None, verbose=False):
    files_names = os.listdir(dir_path)
    if extensions:
        files_names = [file_name for file_name in files_names if matches_any_extension(file_name, extensions)]
    if exclude_files:
        removed = 0
        for exclude_file_name in exclude_files:
            if exclude_file_name in files_names:
                files_names.remove(exclude_file_name)
                removed += 1
            if verbose:
                print(f"  Number of files excluded: {removed}", end="\r", flush=True)
    if verbose:
        print(f"\n  Found {len(files_names)} files with one of these extensions {extensions} in '{dir_path}'")
    return files_names


@timer_wrapper
def load_data(data_dir, img_key, steering_key, throttle_key, extensions=None, exclude_files=None, verbose=False):
    print("# Loading data")
    json_files = list_dir(data_dir, extensions=extensions, exclude_files=exclude_files, verbose=verbose)
    json_keys = (img_key, steering_key, throttle_key)
    img_data, actions = [], []
    pb = ProgressBar(total_iter=len(json_files), display_text="  Loading images", display_interval=10)
    for json_file in json_files:
        json_data = load_json(os.path.join(data_dir, json_file), include_keys=json_keys)
        throttle = json_data[throttle_key] or 0.0
        steering = json_data[steering_key] or 0.0
        # TODO: Ignore data where the car was in rest
        # TODO: BGR2RGB conversion only required for visual purpose. Maybe use YUV format instead
        img_path = os.path.join(data_dir, json_data[img_key])
        img = load_img(img_path, cvt_color=cv2.COLOR_BGR2RGB)
        action = (steering, throttle)
        img_data.append(img)
        actions.append(action)
        pb.step(1)
    return np.array(img_data), np.array(actions)


def data_generator(images, actions, epochs=1, batch_size=10, preprocessors=None):
    images, actions = shuffle(images, actions)
    total_samples = len(images)
    _preprocessors = [None]
    if preprocessors:
        assert isinstance(preprocessors, (list, tuple, set)), \
            f"Preprocessors should be an iterator, not {type(preprocessors)}"
        for processor in preprocessors:
            vectorized_func = np.vectorize(processor, otypes=[np.ndarray, np.ndarray], cache=True)
            _preprocessors.append(vectorized_func)
    p_bar = ProgressBar(total_iter=len(actions)*len(_preprocessors), change_line_at_reset=False,
                        display_interval=batch_size)
    for epoch in range(epochs):
        p_bar.set_display_text(f"  Epoch {epoch+1:<2}")
        for offset in range(0, total_samples, batch_size):
            for processor in _preprocessors:
                batch_images = np.array(images[offset:offset+batch_size])
                batch_actions = np.array(actions[offset:offset+batch_size])
                if processor:
                    batch_images, batch_actions = processor(batch_images, batch_actions)
                # TODO: Normalization may differ for YUV format
                batch_images = normalize_images(batch_images)
                p_bar.step(len(batch_actions))
                yield batch_images, batch_actions


def clip_steering_tf(value):
    import tensorflow as tf
    return tf.clip_by_value(value, STEER_LIMIT_LEFT, STEER_LIMIT_RIGHT)


def clip_steering(value):
    return np.clip(value, STEER_LIMIT_LEFT, STEER_LIMIT_RIGHT)


def clip_throttle_tf(value):
    import tensorflow as tf
    return tf.clip_by_value(value, THROTTLE_MIN, THROTTLE_MAX)


def clip_throttle(value):
    return np.clip(value, THROTTLE_MIN, THROTTLE_MAX)


def parse_args(mode):
    """
    Parse arguments from command line
    returns:
         argparse.ArgumentParser : Parsed arguments
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--version", help="DonkeyNet version", type=int,
                            choices=(0, 1, 2), default=1)
    if mode.lower() == "train":
        arg_parser.add_argument("--data_dirs", help="Data directories", nargs="+", required=True)
        arg_parser.add_argument("--epochs", help="Number of epochs", type=int, default=20)
        arg_parser.add_argument("--img_key", help="Key value for image array in JSON data",
                                type=str, default="cam/image_array")
        arg_parser.add_argument("--steering_key", help="Key value for steering in JSON data",
                                type=str, default="user/angle")
        arg_parser.add_argument("--throttle_key", help="Key value for throttle in JSON data",
                                type=str, default="user/throttle")
        arg_parser.add_argument("--save_model_path", help="Path to save for trained model",
                                type=str, default=None)
        arg_parser.add_argument("--retrain_model", help="Path of model to retrain",
                                type=str, default=None)
        arg_parser.add_argument("--verbose", help="Verbosity", dest="verbose", action="store_true", default=False)
        arg_parser.add_argument("--add_flip", help="Add flip pre-processing", dest="add_flip", action="store_true",
                                default=False)
        arg_parser.add_argument("--add_blur", help="Add blur pre-processing", dest="add_blur", action="store_true",
                                default=False)
        _args = arg_parser.parse_args()
        if _args.save_model_path is None:
            _args.save_model_path = os.path.join("models", f"DonkeyNetV{_args.version}Model")
        os.makedirs(_args.save_model_path, exist_ok=True)
    else:
        arg_parser.add_argument("--model_path", help="Model checkpoint directory", type=str)
        arg_parser.add_argument("--sim_rate", help="Simulation rendering frequency in Hz", type=int, default=20)
        arg_parser.add_argument("--throttle", help="Car throttle value", type=float, default=0.2)
        arg_parser.add_argument("--recording_path", help="Path to save any recording data", type=str, default=None)
        donkey_gym_envs = ["donkey-generated-roads-v0", "donkey-warehouse-v0",
                           "donkey-avc-sparkfun-v0", "donkey-generated-track-v0"]
        arg_parser.add_argument("--env_name", help="Donkey Gym environment name", type=str,
                                default="donkey-generated-track-v0",
                                choices=donkey_gym_envs)
        _args = arg_parser.parse_args()
        if _args.recording_path is not None:
            os.makedirs(_args.recording_path, exist_ok=True)
    return _args


class ContextManagerWrapper:
    # A class wrapper to convert normal object into context managers
    def __init__(self, obj, exit_method):
        self.obj = obj
        self.exit_method = getattr(obj, exit_method)

    def __enter__(self):
        return self.obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_method()
