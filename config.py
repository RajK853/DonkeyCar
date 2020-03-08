# Simulation variables
SIM_PATH = r"/home/rajk/Machine_Learning/DonkeySimLinux/donkey_sim.x86_64"
SIM_RATE = 60
# Steering variables
STEERING_CHANNEL = 0
STEERING_LEFT_PWM = 110
STEERING_RIGHT_PWM = 260
# Throttle variables
THROTTLE_CHANNEL_LEFT = 3
THROTTLE_FORWARD_PWM_LEFT = 4000
THROTTLE_STOPPED_PWM_LEFT = 100
THROTTLE_REVERSE_PWM_LEFT = 0
# Camera variables
CAM_TYPE = "web_cam"
IMAGE_RESOLUTION = (IMAGE_H, IMAGE_W, IMAGE_D) = (120, 160, 3)
# Joystick
CONTROLLER_TYPE = "web_ctr"              # web_ctr|ps3
JOYSTICK_THROTTLE_DIR = -1.0,
JOYSTICK_MAX_THROTTLE = 1.0,
JOYSTICK_STEERING_SCALE = 1.0,
AUTO_RECORD_ON_THROTTLE = False,
JOYSTICK_DEADZONE = 0.0
# Train variables
INCLUDE_THROTTLE = True
INCLUDE_SENSORS = False
SENSOR_NUM = 7
IMG_KEY = "cam/image_array"
STEERING_KEY = "steering"
THROTTLE_KEY = "throttle"
SENSOR_KEYS = ["sensor_fr", "sensor_mr", "sensor_br", "sensor_bm", "sensor_bl", "sensor_ml", "sensor_fl"]
# Ultrasonic sensors trigger and echo pins in gpio board mode
SENSOR_PINS = [(40, 42), (40, 42), (40, 42), (40, 42), (40, 42), (40, 42), (40, 42)]
SEQUENCE_LENGTH = 1
