import RPi.GPIO as GPIO

LOW = GPIO.LOW
HIGH = GPIO.HIGH
OUT = GPIO.OUT


class ThrottleGPIOController:
    def __init__(self, pwm, ain1, ain2, stby):
        self.pins = self.pwm, self.ain1, self.ain2, self.stby = (pwm, ain1, ain2, stby)
        GPIO.setmode(GPIO.BOARD)
        self._set_all(HIGH)

    def _set_all(self, value):
        assert value in (LOW, HIGH), f"Invalid pin value; received {value}"
        for pin in self.pins:
            GPIO.setup(pin, value)

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
        self._set_all(LOW)
