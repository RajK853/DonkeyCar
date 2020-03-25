import os
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf_v1
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.util import deprecation

import config
from src.arg_parser import parse_args
from src.models import get_donkey_net_func
from src.preprocessors import get_preprocessor
from src.utils import load_multiple_data, copy_attributes
from src.data_generators import DataGenerator, DataGeneratorClassifier

# Tensorflow session configuration
tf_v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
deprecation._PRINT_DEPRECATION_WARNINGS = False
GPU_OPTIONS = tf_v1.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
TF_CONFIG = tf_v1.ConfigProto(gpu_options=GPU_OPTIONS)
tf_v1.keras.backend.set_session(tf_v1.Session(config=TF_CONFIG))


def load_data(data_type):
    """
    This local function prepares arguments and loads the data based on given data_type
    :param data_type: (str) "train", "test" or "validation"
    :return: (dict) Dictionary with image, sensor and action data
    """
    kwargs = {"data_dir": os.path.join(args.data_dir, data_type),
              "config": config,
              "force_process": args.force_process,
              "data_type": data_type}
    return load_multiple_data(**kwargs)


def init_model(_args):
    img_shape = config.IMAGE_RESOLUTION
    sensor_shape = (config.SENSOR_NUM,)
    if config.sequence_length > 1:
        img_shape = (config.sequence_length,) + img_shape
        sensor_shape = (config.sequence_length,) + sensor_shape
    func_name = f"donkey_{_args.model_type}_v{_args.version}"
    donkey_net = get_donkey_net_func(func_name)
    model = donkey_net(img_shape=img_shape, sensor_shape=sensor_shape, lr=_args.lr)
    return model


def init_data_generators(_args):
    dataset = {data_type: load_data(data_type) for data_type in config.DATA_TYPES}
    # Add preprocessors for training dataset
    print("# Adding preprocessors")
    preprocessors = []
    for p in _args.preprocessors:
        p_func = get_preprocessor(p)
        preprocessors.append(p_func)
        print(f"  Added {p} preprocessor!")
    dataset["train"]["preprocessors"] = preprocessors
    dataset["train"]["batch_size"] = dataset["validation"]["batch_size"] = _args.batch_size
    dataset["test"]["batch_size"] = 1
    if _args.model_type == "model":
        gen_func = DataGenerator
    else:
        gen_func = DataGeneratorClassifier
    gen_dict = {data_type: gen_func(**dataset[data_type], sequence_len=config.sequence_length)
                       for data_type in config.DATA_TYPES}
    return gen_dict


def display_test_result(test_result):
    if isinstance(test_result, list):
        num_metrics = len(test_result)
        keys = list(train_result.history.keys())[:num_metrics]
        data_dict = dict(zip(keys, test_result))
        print(f"Test result: {data_dict}")
    else:
        print(f"Test loss: {test_result:.4f}")


def plot_training_result(train_history, plt_dump_path=None):
    for metric in train_history.values():
        plt.plot(metric)
    plt.title("Training result")
    plt.xlabel("Epochs")
    plt.legend(list(train_history.keys()))
    if plt_dump_path is not None:
        plt.savefig(plt_dump_path)
        print(f"- Graphs saved to {plt_dump_path}")
    plt.show()


if __name__ == "__main__":
    args = parse_args(mode="train")
    copy_attributes(("predict_throttle", "using_sensors", "sequence_length"), src_obj=args, dst_obj=config)
    model = init_model(args)
    data_generators = init_data_generators(args)
    print(f"\n# Training the {args.model_type}!")
    # train_callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, restore_best_weights=True)]
    train_callbacks = []
    train_result = model.fit(data_generators["train"], validation_data=data_generators["validation"],
                             epochs=args.epochs, callbacks=train_callbacks)
    print(f"\n# Evaluating the {args.model_type}")
    test_loss = model.evaluate(data_generators["test"])
    display_test_result(test_loss)
    plt_save_path = os.path.join(args.model_path, f"Train_result-{args.model_type}.jpg")
    plot_training_result(train_result.history, plt_dump_path=plt_save_path)
    print(f"\n# Saving the {args.model_type}")
    model_file_path = os.path.join(args.model_path, f"{args.model_type}.h5")
    model.save(model_file_path)
    print(f"  {args.model_type.capitalize()} saved to '{model_file_path}'")
