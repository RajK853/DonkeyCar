import os
import cv2
import argparse
import numpy as np
import pandas as pd
import json
from time import time
from functools import wraps
from sklearn.utils import shuffle

from .progress_bar import ProgressBar

STEER_LIMIT_LEFT = -1.0
STEER_LIMIT_RIGHT = 1.0
THROTTLE_MIN = -1.0
THROTTLE_MAX = 1.0


def timer_wrapper(func):
    @wraps(func)
    def wrapper_func(*args, **kwargs):
        t0 = time()
        result = func(*args, **kwargs)
        dt = time()-t0
        print(f"- Function '{func.__name__}' took {dt:.3f} seconds!\n")
        return result
    return wrapper_func


def crop_img(img, crop_dim):
    assert all(isinstance(crop, slice) for crop in crop_dim), f"Crop_dim should be an slice, not {type(crop_dim)}"
    img = img[crop_dim]
    return img


def flip_img(img, action, sensor_data=None):
    flipped_img = cv2.flip(img, 1)
    # Make changes in the copy of action array
    action = action.copy()
    action[0] *= -1              # Flip steering direction
    if sensor_data is not None:
        sensor_data = np.fliplr(sensor_data)
    # TODO: Swap right and left ultrasonic sensor readings
    # TODO: Then return sensor data too
    return flipped_img, action, sensor_data


def blur_img(img, action, sensor_data=None):
    img = cv2.GaussianBlur(img, (9, 9), 0)
    return img, action, sensor_data


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


def list_dir(dir_path, extensions=None, exclude_files=None):
    files_names = os.listdir(dir_path)
    if extensions:
        files_names = [file_name for file_name in files_names if matches_any_extension(file_name, extensions)]
    if exclude_files:
        removed = 0
        for exclude_file_name in exclude_files:
            if exclude_file_name in files_names:
                files_names.remove(exclude_file_name)
                removed += 1
            print(f"  Number of files excluded: {removed}", end="\r", flush=True)
    print(f"\n  Found {len(files_names)} files with one of these extensions {extensions} in '{dir_path}'")
    return files_names


@timer_wrapper
def process_json_data(data_dir, include_keys, exclude_files=None, force_process=False):
    csv_file = os.path.join(data_dir, "processed_data.csv")
    if (not force_process) and os.path.exists(csv_file):
        print(f"- Processed file already present at {data_dir}!")
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame()
        json_files = list_dir(data_dir, extensions=[".json"], exclude_files=exclude_files)
        p_bar = ProgressBar(total_iter=len(json_files), display_text="  Processing data", display_interval=10)
        for json_file in json_files:
            json_path = os.path.join(data_dir, json_file)
            data = load_json(json_path, include_keys=include_keys)
            df = df.append(data, ignore_index=True)
            p_bar.step(1)
        df.to_csv(csv_file, index=False)
        print(f"- Data processed successfully and saved to '{csv_file}'")
    return df


@timer_wrapper
def load_data(data_dir, config, force_process=False):
    img_key = config.IMG_KEY
    steering_key = config.STEERING_KEY
    throttle_key = config.THROTTLE_KEY
    sensor_keys = config.SENSOR_KEYS
    input_keys = [img_key]
    action_keys = [steering_key]
    if config.SENSOR_NUM > 0:
        input_keys.extend(sensor_keys)
    if config.INCLUDE_THROTTLE:
        action_keys.append(throttle_key)
    include_keys = input_keys + action_keys
    df = process_json_data(data_dir, include_keys=include_keys, exclude_files=["meta.json"],
                           force_process=force_process)
    print(f"# Loading {len(df)} data...")
    df[img_key] = df[img_key].apply(lambda x: os.path.join(data_dir, x))
    vec_func = np.vectorize(load_img, otypes=[np.ndarray], cache=False)
    images = vec_func(df[img_key])
    actions = df[action_keys].values
    if config.INCLUDE_SENSORS:
        sensor_data = df[sensor_keys].values
    else:
        sensor_data = []
    return images, sensor_data, actions


def data_generator(images, actions, epochs=1, batch_size=10, preprocessors=None, sensor_data=None, sequence=1):
    images = np.array(images)
    actions = np.array(actions)
    actions = actions.reshape((*actions.shape, 1))
    sensor_data_present = sensor_data is not None and (len(sensor_data) > 0)
    if sensor_data_present:
        sensor_data = np.array(sensor_data)
    total_samples = len(images)
    _preprocessors = [blur_img]
    if preprocessors:
        assert isinstance(preprocessors, (list, tuple, set)), \
            f"Preprocessors should be an iterator, not {type(preprocessors)}"
        _preprocessors.extend(preprocessors)
    indexes = np.array(range(total_samples))
    make_sequence = sequence > 1
    start_index = sequence-1 if make_sequence else 0
    p_bar = ProgressBar(total_iter=(total_samples-start_index)*len(_preprocessors), change_line_at_reset=False,
                        display_interval=batch_size)
    for epoch in range(epochs):
        indexes = shuffle(indexes)
        p_bar.set_display_text(f"  Epoch {epoch+1:<2}")

        for offset in range(start_index, total_samples, batch_size):
            batch_indexes = indexes[offset:offset + batch_size]
            actual_batch_size = len(batch_indexes)
            if make_sequence:
                batch_indexes = np.hstack([batch_indexes] + [batch_indexes.copy() - i for i in range(1, sequence)])
            batch_images = images[batch_indexes]
            batch_actions = actions[batch_indexes]
            if sensor_data_present:
                batch_sensor_data = sensor_data[batch_indexes]
            else:
                batch_sensor_data = [None]*len(batch_indexes)
            for processor in _preprocessors:
                # TODO: Improve performance with vectorization?
                batch_results = [processor(img, act, sen)
                                 for img, act, sen in zip(batch_images, batch_actions, batch_sensor_data)]
                _batch_images, _batch_actions, _batch_sensor_data = zip(*batch_results)
                _batch_images = np.array(_batch_images)
                _batch_actions = np.array(_batch_actions[:actual_batch_size])
                _batch_sensor_data = np.array(_batch_sensor_data)
                # TODO: Normalization may differ for YUV format
                _batch_images = normalize_images(_batch_images)
                if make_sequence:
                    _batch_images = _batch_images.reshape((actual_batch_size, sequence, *_batch_images.shape[1:]))
                    _batch_sensor_data = _batch_sensor_data.reshape((actual_batch_size, sequence, *_batch_sensor_data.shape[1:]))
                p_bar.step(actual_batch_size)
                if sensor_data_present:
                    yield [_batch_images, _batch_sensor_data], _batch_actions
                else:
                    yield [_batch_images], _batch_actions


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
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--version", help="DonkeyNet version", type=int)
    if mode.lower() == "train":
        arg_parser.add_argument("--train_data_dirs", help="Train data directories", nargs="+", required=True)
        arg_parser.add_argument("--test_data_dirs", help="Test data directories", nargs="+", required=True)
        arg_parser.add_argument("--epochs", help="Number of epochs", type=int, default=20)
        arg_parser.add_argument("--batch_size", help="Batch size while training", type=int, default=126)
        arg_parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
        arg_parser.add_argument("--img_key", help="Key value for image array in JSON data",
                                type=str, default="cam/image_array")
        arg_parser.add_argument("--steering_key", help="Key value for steering in JSON data",
                                type=str, default="steering")
        arg_parser.add_argument("--throttle_key", help="Key value for throttle in JSON data",
                                type=str, default="throttle")
        arg_parser.add_argument("--save_model_path", help="Path to save for trained model",
                                type=str, default=None)
        arg_parser.add_argument("--retrain_model", help="Path of model to retrain",
                                type=str, default=None)
        arg_parser.add_argument("--add_flip", help="Add flip pre-processing", dest="add_flip", action="store_true",
                                default=False)
        arg_parser.add_argument("--force_process", help="Force data processing", dest="force_process",
                                action="store_true", default=False)
        _args = arg_parser.parse_args()
        if _args.save_model_path is None:
            _args.save_model_path = os.path.join("models", f"DonkeyNetV{_args.version}Model")
        os.makedirs(_args.save_model_path, exist_ok=True)
    else:
        arg_parser.add_argument("--model_path", help="Model checkpoint directory", type=str)
        arg_parser.add_argument("--sim_rate", help="Simulation rendering frequency in Hz", type=int, default=60)
        arg_parser.add_argument("--throttle", help="Car throttle value", type=float, default=0.3)
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


def get_camera(cam_type):
    if cam_type == "pi_cam":
        from donkeycar.parts.camera import PiCamera as Camera
    elif cam_type == "web_cam":
        from donkeycar.parts.camera import Webcam as Camera
    else:
        raise TypeError(f"Invalid camera type: {cam_type}")
    return Camera
