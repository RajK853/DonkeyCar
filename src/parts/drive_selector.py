from . import BasePart


class DriveSelector(BasePart):

    def run(self, user_steering, user_throttle, pilot_steering, pilot_throttle, mode):
        if mode == "user":
            auto_mode = False
            steering, throttle = user_steering, user_throttle
        elif mode == "local_angle":
            auto_mode = True
            steering, throttle = pilot_steering, user_throttle
        else:
            auto_mode = True
            steering, throttle = pilot_steering, pilot_throttle
        if steering is None:
            steering = 0.0
        if throttle is None:
            throttle = 0.0
        return steering, throttle, auto_mode
