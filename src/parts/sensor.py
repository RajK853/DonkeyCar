import RPi.GPIO as GPIO
from time import time, sleep

LOW = GPIO.LOW
HIGH = GPIO.HIGH
OUT = GPIO.OUT
IN = GPIO.IN
MAX_DISTANCE = 3.6  # Unit: [m]


class UltraSonic(object):
    def __init__(self, trig_pin, echo_pin, trig_time=1e-6, mode=GPIO.BOARD):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.trig_time = trig_time
        self.distance = MAX_DISTANCE
        GPIO.setmode(mode)
        self.setup()
        self.running = True

    def setup(self):
        GPIO.setup(self.trig_pin, OUT)
        GPIO.setup(self.echo_pin, IN)

    def gen_pulse(self):
        GPIO.output(self.trig_pin, HIGH)
        sleep(self.trig_time)
        GPIO.output(self.trig_pin, LOW)

    def calculate_distance(self):
        self.gen_pulse()
        echo_pin = self.echo_pin
        t0 = t1 = 0.0
        while GPIO.input(echo_pin) == 0:
            t0 = time()
        while GPIO.input(echo_pin) == 1:
            t1 = time()
        dt = t1 - t0
        distance = 171.5 * dt  # Equals to (343 [m/s] * dt [s])/2
        self.distance = min(distance, MAX_DISTANCE)

    def update(self):
        calculate_distance = self.calculate_distance
        while self.running:
            calculate_distance()

    def run_threaded(self):
        return self.distance

    def run(self):
        self.calculate_distance()
        return self.distance

    def shutdown(self):
        self.running = False
        GPIO.cleanup()
