class DriveSelector:
    def __init__(self, pid_ctr=None):
        self.pid_ctr = pid_ctr

    def run(self, user_steering, user_throttle, pilot_steering, pilot_throttle, mode):
        auto_mode = (mode != "user")
        steering, throttle = (pilot_steering, pilot_throttle) if auto_mode else (user_steering, user_throttle)
        if steering is None:
            steering = 0.0
        if throttle is None:
            throttle = 0.0
        if mode == "local" and self.pid_ctr is not None:
            steering = self.pid_ctr.run(steering)
        return steering, throttle, auto_mode
