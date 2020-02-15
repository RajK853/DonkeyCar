import numpy as np
from time import time


class PIDController(object):
    def __init__(self, P, I, D, max_value, min_value, store_n_errors=30, target_value=0.0):
        self.P = P
        self.I = I
        self.D = D
        self.max_value = max_value
        self.min_value = min_value
        self._target_value = target_value
        self.max_n_errors = store_n_errors
        self.previous_errors = []
        self.last_exec_time = None

    @property
    def target_value(self):
        return self._target_value

    @target_value.setter
    def target_value(self, value):
        self._target_value = value

    def add_error(self, error):
        if len(self.previous_errors) >= self.max_n_errors:
            self.previous_errors.pop(0)
        self.previous_errors.append(error)

    @property
    def dt(self):
        current_time = time()
        _dt = current_time - self.last_exec_time
        self.last_exec_time = current_time
        return _dt

    def run(self, value):
        error_p = (self._target_value - value)
        self.add_error(error_p)
        if self.last_exec_time is None:
            self.last_exec_time = time()
            error_d = 0
            error_i = 0
        else:
            dt = self.dt
            error_i = np.mean(self.previous_errors) * dt
            error_d = (self.previous_errors[-1] - self.previous_errors[-2])/dt
        out_values = (self.P * error_p, self.I * error_i, self.D * error_d)
        # print(f"PID output values: {out_values}")
        return np.clip(sum(out_values), self.min_value, self.max_value)
