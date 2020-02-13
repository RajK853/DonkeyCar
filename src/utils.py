import os
import cv2
import json
import argparse
import numpy as np
from sklearn.utils import shuffle
from .progress_bar import ProgressBar

STEER_LIMIT_LEFT = -1.0
STEER_LIMIT_RIGHT = 1.0
THROTTLE_MIN = 0.0
THROTTLE_MAX = 5.0


def crop_img(img, crop_dim):
    assert all(isinstance(crop, slice) for crop in crop_dim), f"Crop_dim should be an slice, not {type(crop_dim)}"
    img = img[crop_dim]
    return img


def load_img(img_file, cvt_color=None, crop_dim=None):
    img = cv2.imread(img_file)
    if crop_dim:
        img = crop_img(img, crop_dim)
    if cvt_color:
        img = cv2.cvtColor(img, cvt_color)
    return img


def load_json(json_file, include_keys=None):
    with open(json_file, "r") as json_f:
        json_data = json.load(json_f)
    if include_keys:
        temp_json_data = {key: json_data[key] for key in include_keys}
        json_data = temp_json_data
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
                print(f"Number of files excluded: {removed}", end="\r", flush=True)
    if verbose:
        print(f"\nFound {len(files_names)} files with one of these extensions {extensions} in '{dir_path}'")
    return files_names


def load_data(data_dir, img_key, steering_key, throttle_key, extensions=None, exclude_files=None, crop_dim=None, verbose=False):
    json_files = list_dir(data_dir, extensions=extensions, exclude_files=exclude_files, verbose=verbose)
    json_keys = (img_key, steering_key, throttle_key)
    img_data, actions = [], []
    for json_file in json_files:
        json_data = load_json(os.path.join(data_dir, json_file), include_keys=json_keys)
        # Ignore data where the car was in rest
        if json_data[throttle_key] == 0.0:
            continue
        img_path = os.path.join(data_dir, json_data[img_key])
        # TODO: BGR2RGB conversion only required for visual purpose. Maybe use YUV format instead
        img = load_img(img_path, cvt_color=cv2.COLOR_BGR2RGB, crop_dim=crop_dim)
        action = np.array((json_data[steering_key], json_data[throttle_key]))
        img_data.append(img)
        actions.append(action)
    return np.array(img_data), np.array(actions)


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


def data_generator(images, actions, epochs=1, batch_size=10, preprocessors=None):
    images, actions = shuffle(images, actions)
    total_samples = len(images)
    _preprocessors = [None]
    if preprocessors:
        assert isinstance(preprocessors, (list, tuple, set)), \
            f"Preprocessors should be an iterator, not {type(preprocessors)}"
        _preprocessors.extend(preprocessors)
    p_bar = ProgressBar(total_iter=len(actions)*len(_preprocessors), change_line_at_reset=False)
    for epoch in range(epochs):
        p_bar.set_display_text(f"Epoch {epoch+1:<2}")
        for processor in _preprocessors:
            for offset in range(0, total_samples, batch_size):
                batch_images = images[offset:offset+batch_size]
                batch_actions = actions[offset:offset+batch_size]
                if processor is not None:
                    batch_results = [processor(img, act) for img, act in zip(batch_images, batch_actions)]
                    batch_images, batch_actions = zip(*batch_results)
                    batch_images = np.array(batch_images)
                    batch_actions = np.array(batch_actions)
                # TODO: Normalization may differ for YUV format
                batch_images = normalize_images(batch_images)
                p_bar.step(len(batch_actions))
                yield batch_images, batch_actions


def clip_steering(value, mode="tf"):
    if mode == "tf":
        import tensorflow as tf
        return tf.clip_by_value(value, STEER_LIMIT_LEFT, STEER_LIMIT_RIGHT)
    return np.clip(value, STEER_LIMIT_LEFT, STEER_LIMIT_RIGHT)


def clip_throttle(value, mode="tf"):
    if mode == "tf":
        import tensorflow as tf
        return tf.clip_by_value(value, THROTTLE_MIN, THROTTLE_MAX)
    return np.clip(value, THROTTLE_MIN, THROTTLE_MAX)


def parse_args():
    """
    Parse arguments from command line
    returns:
         argparse.ArgumentParser : Parsed arguments
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", help="Data directory", type=str)
    arg_parser.add_argument("--epochs", help="Number of epochs", type=int, default=20)
    arg_parser.add_argument("--img_key", help="Key value for image array in JSON data",
                            type=str, default="cam/image_array")
    arg_parser.add_argument("--steering_key", help="Key value for steering in JSON data",
                            type=str, default="user/angle")
    arg_parser.add_argument("--throttle_key", help="Key value for throttle in JSON data",
                            type=str, default="user/throttle")
    arg_parser.add_argument("--save_model_path", help="Path to save for trained models",
                            type=str, default=None)
    arg_parser.add_argument("--verbose", help="Verbosity", dest="verbose", action="store_true")
    arg_parser.set_defaults(verbose=False)
    _args = arg_parser.parse_args()
    if _args.save_model_path is None:
        _args.save_model_path = os.path.join("models", "model.chkpt")
    return _args
