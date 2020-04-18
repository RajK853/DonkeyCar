import RPi.GPIO as GPIO
from time import time, sleep
from numpy.random import uniform

from . import BasePart

LOW = GPIO.LOW
HIGH = GPIO.HIGH
OUT = GPIO.OUT
IN = GPIO.IN
MAX_DISTANCE = 3.6  # Unit: [m]


class FakeSensor(BasePart):
    def __init__(self, name, max_distance=MAX_DISTANCE):
        self.name = name
        self.running = True
        self.max_distance = max_distance
        self._distance = max_distance

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        dist = min(value, self.max_distance)
        dist = round(dist, 4)
        self._distance = dist

    def calculate_distance(self):
        self.distance = uniform(0.6, self.max_distance)

    def update(self):
        while self.running:
            self.calculate_distance()

    def run_threaded(self):
        return self.distance

    def run(self):
        self.calculate_distance()
        distance = self.distance
        return distance

    def shutdown(self):
        self.running = False


class UltraSonic(FakeSensor):
    def __init__(self, name, trig_pin, echo_pin, max_distance, trig_time=10e-6, timeout_t=0.2, mode=GPIO.BOARD):
        super(UltraSonic, self).__init__(name, max_distance)
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.trig_time = trig_time
        self.timeout_t = timeout_t
        GPIO.setmode(mode)
        self.setup()

    def setup(self):
        GPIO.setup(self.trig_pin, OUT)
        GPIO.setup(self.echo_pin, IN)

    def gen_pulse(self):
        sleep(2*self.trig_time)
        GPIO.output(self.trig_pin, HIGH)
        sleep(self.trig_time)
        GPIO.output(self.trig_pin, LOW)

    def calculate_distance(self):
        t0 = t1 = 0.0
        max_time = time() + self.timeout_t
        self.gen_pulse()
        while GPIO.input(self.echo_pin) == 0:
            t0 = time()
            if t0 > max_time:
                print(f"Timeout in {self.name}. Returned value: {self.distance}")
                return
        while GPIO.input(self.echo_pin) == 1:
            t1 = time()
            if t1 > max_time:
                print(f"Timeout in {self.name}. Returned value: {self.distance}")
                return
        dt = t1 - t0
        distance = 171.5 * dt  # Equals to (343 [m/s] * dt [s])/2
        self.distance = distance

    def shutdown(self):
        self.running = False
        GPIO.cleanup()


