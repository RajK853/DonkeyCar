import RPi.GPIO as GPIO

LOW = GPIO.LOW
HIGH = GPIO.HIGH
OUT = GPIO.OUT


class ThrottleGPIOController:
    def __init__(self, pwm, ain1, ain2, stby):
        self.pins = self.pwm, self.ain1, self.ain2, self.stby = (pwm, ain1, ain2, stby)
        GPIO.setmode(GPIO.BOARD)
        self.setup()

    def setup(self):
        self._set_all(OUT, set_func=GPIO.setup)

    def _set_all(self, value, set_func=GPIO.output):
        assert  set_func in (GPIO.setup, GPIO.output), f"Invalid Rpi.GPIO setup function: {set_func.__name__} received!"
        assert value in (LOW, HIGH, OUT), f"Invalid pin value: {value} received!"
        for pin in self.pins:
            set_func(pin, value)

    def run(self, throttle):
        if throttle > 0:
            GPIO.output(self.ain2, LOW)
            GPIO.output(self.ain1, HIGH)
        elif throttle < 0:
            GPIO.output(self.ain2, HIGH)
            GPIO.output(self.ain1, LOW)
            throttle *= -1
        return throttle

    def close(self):
        self._set_all(LOW, set_func=GPIO.output)
