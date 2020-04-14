import os
import argparse


def parse_args(mode):
    arg_parser = argparse.ArgumentParser()
    if mode.lower() == "train":
        arg_parser.add_argument("--version", help="DonkeyNet version", type=int)
        arg_parser.add_argument("--epochs", help="Number of epochs", type=int, default=20)
        arg_parser.add_argument("--model_type", help="Classifier or normal Model", type=str, default="model")
        arg_parser.add_argument("--data_dir", help="Directory with train, test and validation data", required=True)
        arg_parser.add_argument("--batch_size", help="Batch size while training", type=int, default=126)
        arg_parser.add_argument("--model_path", help="Path to save for trained model", type=str, default=None)
        arg_parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
        # TODO: Add preprocessors instead
        arg_parser.add_argument("--preprocessors", help="Preprocessor names", nargs="+", default=None)
        arg_parser.add_argument("--force_process", help="Force data processing", dest="force_process",
                                action="store_true", default=False)
        _args = arg_parser.parse_args()
        if _args.preprocessors is None:
            _args.preprocessors = []
        if _args.model_type == "classifier":
            _args.predict_throttle = True
        if _args.model_path is None:
            _args.model_path = os.path.join("models", f"DonkeyNetV{_args.version}Model")
        os.makedirs(_args.model_path, exist_ok=True)
    else:
        arg_parser.add_argument("--model_path", help="Model checkpoint directory", type=str, default=None)
        arg_parser.add_argument("--classifier_model_path", help="Classifier Model checkpoint directory", type=str,
                                default=None)
        arg_parser.add_argument("--sim_rate", help="Simulation rendering frequency in Hz", type=int, default=60)
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
