import os
import cv2
import numpy as np
import pandas as pd
import json
import tensorflow.compat.v1 as tf_v1

from .progress_bar import ProgressBar
from .wrappers import timer_wrapper

STEER_LIMIT_LEFT = -1.0
STEER_LIMIT_RIGHT = 1.0
THROTTLE_MIN = -1.0
THROTTLE_MAX = 1.0


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
    print(f"  Found {len(files_names)} files with one of these extensions {extensions} in '{dir_path}'")
    return files_names


def process_json_data(data_dir, include_keys=None, exclude_files=None, force_process=False):
    csv_file = os.path.join(data_dir, "processed_data.csv")
    if (not force_process) and os.path.exists(csv_file):
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


def load_data(data_dir, config, force_process=False):
    # Cache variables for faster access
    img_key = config.IMG_KEY
    steering_key = config.STEERING_KEY
    throttle_key = config.THROTTLE_KEY
    sensor_keys = config.SENSOR_KEYS
    # Configure action keys
    action_keys = [steering_key]
    if config.predict_throttle:
        action_keys.append(throttle_key)
    df = process_json_data(data_dir, exclude_files=["meta.json"],
                           force_process=force_process)
    print(f"  Found {len(df)} data in '{data_dir}'")
    # Convert relative image paths to absolute paths
    df[img_key] = df[img_key].apply(lambda x: os.path.join(data_dir, x))
    vec_func = np.vectorize(load_img, otypes=[np.ndarray], cache=False)
    images = vec_func(df[img_key])
    actions = df[action_keys].values
    if config.using_sensors:
        sensor_data = df[sensor_keys].values
    else:
        sensor_data = []
    return np.array(images), np.array(sensor_data), np.array(actions)


@timer_wrapper
def load_multiple_data(data_dir, config, force_process=False, data_type=None):
    data_dict = {"images": [], "actions": [], "sensors": []}
    if data_type is not None:
        print(f"# Loading {data_type} data:")
    for root_dir, _, files in os.walk(data_dir):
        if files:
            images, sensor_data, actions = load_data(root_dir, config, force_process)
            if not config.predict_throttle:
                actions = actions[:, 0]
            actions = actions.reshape(*actions.shape, 1)
            data_dict["images"].extend(images)
            data_dict["actions"].extend(actions)
            data_dict["sensors"].extend(sensor_data)
    return data_dict


def clip_steering_tf(value):
    import tensorflow.compat.v1 as tf_v1
    return tf_v1.clip_by_value(value, STEER_LIMIT_LEFT, STEER_LIMIT_RIGHT)


def clip_steering(value):
    return np.clip(value, STEER_LIMIT_LEFT, STEER_LIMIT_RIGHT)


def clip_throttle_tf(value):
    import tensorflow.compat.v1 as tf_v1
    return tf_v1.clip_by_value(value, THROTTLE_MIN, THROTTLE_MAX)


def clip_throttle(value):
    return np.clip(value, THROTTLE_MIN, THROTTLE_MAX)


def copy_attributes(attrs, *, src_obj, dst_obj):
    for attr in attrs:
        attr_value = getattr(src_obj, attr)
        setattr(dst_obj, attr, attr_value)


def get_camera(cam_type):
    if cam_type == "pi_cam":
        from donkeycar.parts.camera import PiCamera as Camera
    elif cam_type == "web_cam":
        from donkeycar.parts.camera import Webcam as Camera
    else:
        raise TypeError(f"Invalid camera type: {cam_type}")
    return Camera


def mixup(*args, alpha=0.2):
    results = []
    batch_size = len(args[0])
    random_indexes = np.random.permutation(batch_size)
    _lambda = np.random.beta(alpha, alpha, size=batch_size)
    for arg in args:
        arg_shape = [1]*len(arg.shape)
        arg_shape[0] = batch_size
        arg_random = arg[random_indexes]
        _lambda = _lambda.reshape(arg_shape)
        arg_convex = _lambda*arg + (1-_lambda)*arg_random
        results.append(arg_convex)
    return results


def load_model(model_path):
    graph = tf_v1.Graph()
    with graph.as_default():
        sess = tf_v1.Session(graph=graph)
        with sess.as_default():
            model = tf_v1.keras.models.load_model(model_path)
            return graph, sess, model
