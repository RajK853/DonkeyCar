import copy
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence
from src.image_processing import normalize_images
from src.utils import mixup


class DataGenerator(Sequence):
    def __init__(self, images, actions, sensors, batch_size, sequence_len=1, preprocessors=None, run_setup=True):
        self.images = np.array(images)
        self.sensors = np.array(sensors)
        self.actions = np.array(actions)
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.preprocessors = preprocessors
        self.preprocessor_idx = 0
        self.preprocessor = None
        self.indexes = None
        self.gen_len = None
        self.make_sequence = None
        self.throttle_in_action = None
        self.sensor_data_present = None
        if run_setup:
            self.setup()

    def setup(self):
        self.indexes = np.arange(len(self.actions))
        self.throttle_in_action = (len(self.actions.shape) > 2)
        self.sensor_data_present = (len(self.sensors) > 0)
        self.make_sequence = (self.sequence_len > 1)
        _preprocessors = [lambda *x, **y: x + tuple(y.values())]
        if self.preprocessors is not None:
            _preprocessors.extend(self.preprocessors)
        self.preprocessors = _preprocessors
        self.preprocessor = self.preprocessors[self.preprocessor_idx]
        self.gen_len = np.math.ceil(len(self.indexes) / self.batch_size)

    def __len__(self):
        return self.gen_len

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        actual_batch_size = len(batch_indexes)
        if self.make_sequence:
            batch_indexes = np.hstack([batch_indexes - i for i in range(0, self.sequence_len)])
        batch_images = copy.deepcopy(self.images[batch_indexes])
        batch_actions = copy.deepcopy(self.actions[batch_indexes])
        if self.sensor_data_present:
            batch_sensor_data = copy.deepcopy(self.sensors[batch_indexes])
        else:
            batch_sensor_data = [None] * len(batch_indexes)
        batch_results = [self.preprocessor(*batch_data)
                         for batch_data in zip(batch_images, batch_actions, batch_sensor_data)]
        batch_images, batch_actions, batch_sensor_data = list(zip(*batch_results))
        batch_images = np.array(batch_images)
        batch_actions = np.array(batch_actions)
        batch_sensor_data = np.array(batch_sensor_data)
        batch_images = normalize_images(batch_images)
        if self.make_sequence:
            batch_images = batch_images.reshape((actual_batch_size, self.sequence_len, *batch_images.shape[1:]))
            batch_sensor_data = batch_sensor_data.reshape(
                (actual_batch_size, self.sequence_len, *batch_sensor_data.shape[1:]))
            batch_actions = batch_actions[0:-1:self.sequence_len]
        batch_output = {}
        if self.throttle_in_action:
            batch_output["steering_output"] = batch_actions[:, 0]
            batch_output["throttle_output"] = batch_actions[:, 1]
        else:
            batch_output["steering_output"] = batch_actions
        if self.sensor_data_present:
            return {"image_input": batch_images, "sensor_input": batch_sensor_data}, batch_output
        else:
            return {"image_input": batch_images}, batch_output

    def on_epoch_end(self):
        self.indexes = shuffle(self.indexes)
        self.preprocessor_idx = (self.preprocessor_idx + 1) % len(self.preprocessors)
        self.preprocessor = self.preprocessors[self.preprocessor_idx]


class DataGeneratorClassifier(DataGenerator):
    def __init__(self, *args, **kwargs):
        run_setup = kwargs.pop("run_setup", True)
        super(DataGeneratorClassifier, self).__init__(*args, **kwargs, run_setup=False)
        assert len(self.actions.shape) > 2, "Actions doesn't include throttle values. Parse --predict_throttle in cmd!"
        throttle = self.actions[:, 1].flat
        is_parked = (throttle <= 0.1)
        self.positive_images = self.images[is_parked]
        self.negative_images = self.images[~is_parked]
        self.positive_sensor_data = self.sensors[is_parked]
        self.negative_sensor_data = self.sensors[~is_parked]
        self.total_positive_samples = len(self.positive_images)
        self.total_negative_samples = len(self.negative_images)
        self.positive_indexes = np.array(range(self.total_positive_samples))
        self.negative_indexes = np.array(range(self.total_negative_samples))
        if run_setup:
            self.setup()

    def setup(self):
        super(DataGeneratorClassifier, self).setup()
        del self.indexes
        min_length = min(self.total_positive_samples, self.total_negative_samples)
        self.gen_len = np.math.ceil(len(self.preprocessors)*min_length / self.batch_size)

    def _process_batch(self, index, indexes, image_data, sensor_data):
        _indexes = indexes[index * self.batch_size:(index + 1) * self.batch_size]
        actual_batch_size = len(_indexes)
        if self.make_sequence:
            _indexes = np.hstack([_indexes - i for i in range(0, self.sequence_len)])
        _images = image_data[_indexes]
        _sensor_data = sensor_data[_indexes]
        return actual_batch_size, _images, _sensor_data

    def __getitem__(self, index):
        batch_result = self._process_batch(index, self.positive_indexes, self.positive_images,
                                           self.positive_sensor_data)
        positive_batch_size, positive_batch_images, positive_batch_sensor_data = batch_result
        batch_result = self._process_batch(index, self.negative_indexes, self.negative_images,
                                           self.negative_sensor_data)
        negative_batch_size, negative_batch_images, negative_batch_sensor_data = batch_result
        batch_images = np.vstack([positive_batch_images, negative_batch_images])
        batch_sensor_data = np.vstack([positive_batch_sensor_data, negative_batch_sensor_data])
        batch_result = [self.preprocessor(img=None, action=None, sensor_data=sen) for sen in batch_sensor_data]
        _, _, batch_sensor_data = list(zip(*batch_result))
        # batch_images = np.array(batch_images)
        batch_sensor_data = np.array(batch_sensor_data)
        batch_labels = np.concatenate(
            [np.ones(positive_batch_size, dtype=int), np.zeros(negative_batch_size, dtype=int)])
        batch_one_hot_labels = np.eye(2, dtype="int32")[batch_labels]
        mixup_features = (batch_sensor_data, batch_one_hot_labels)
        batch_sensor_data, batch_one_hot_labels = mixup(*mixup_features, alpha=1.0)
        # batch_images = normalize_images(batch_images)
        # batch_inputs = {"image_input": batch_images, "sensor_input": batch_sensor_data}
        batch_inputs = {"sensor_input": batch_sensor_data}
        batch_output = {"parked_output": batch_one_hot_labels}
        return batch_inputs, batch_output

    def on_epoch_end(self):
        self.positive_indexes = shuffle(self.positive_indexes)
        self.negative_indexes = shuffle(self.negative_indexes)
        self.preprocessor_idx = (self.preprocessor_idx + 1) % len(self.preprocessors)
        self.preprocessor = self.preprocessors[self.preprocessor_idx]
