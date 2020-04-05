import RPi.GPIO as GPIO
from . import BasePart

LOW = GPIO.LOW
HIGH = GPIO.HIGH
OUT = GPIO.OUT


class WeightedThrottle(BasePart):
    def __init__(self, func=None, min_weight=0.0):
        self.func = lambda x, y: x*y if func is None else func
        self.min_weight = min_weight

    def run(self, throttle, weight, enable):
        if not enable:
            return throttle
        weight = 1.0 if weight is None else weight
        weight = max(weight, self.min_weight)
        result = self.func(throttle, weight)
        return round(result, 4)


class ThrottleGPIOController(BasePart):
    def __init__(self, pwm, ain1, ain2, stby, mode=GPIO.BOARD):
        self.pins = self.pwm, self.ain1, self.ain2, self.stby = (pwm, ain1, ain2, stby)
        GPIO.setmode(mode)
        self.setup()

    def setup(self):
        self._set_all(OUT, call_func=GPIO.setup)

    def _set_all(self, value, call_func=GPIO.output):
        assert call_func in (GPIO.setup, GPIO.output), f"Invalid GPIO call function: {call_func.__name__} received!"
        assert value in (LOW, HIGH, OUT), f"Invalid pin value: {value} received!"
        for pin in self.pins:
            call_func(pin, value)

    def run(self, throttle):
        if throttle > 0:
            GPIO.output(self.ain2, LOW)
            GPIO.output(self.ain1, HIGH)
        elif throttle < 0:
            GPIO.output(self.ain2, HIGH)
            GPIO.output(self.ain1, LOW)
            throttle *= -1
        return throttle

    def shutdown(self):
        self._set_all(LOW, call_func=GPIO.output)
        GPIO.cleanup()
