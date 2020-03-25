# Simulation variables
SIM_PATH = r"/home/rajk/Machine_Learning/DonkeySimLinux/donkey_sim.x86_64"
SIM_RATE = 20
# Dataset variables
DATA_TYPES = ("train", "test", "validation")
# Steering variables
STEERING_CHANNEL = 0
STEERING_LEFT_PWM = 180
STEERING_RIGHT_PWM = 310
# Throttle variables
THROTTLE_CHANNEL_LEFT = 3
THROTTLE_FORWARD_PWM_LEFT = 4000
THROTTLE_STOPPED_PWM_LEFT = 100
THROTTLE_REVERSE_PWM_LEFT = 0
# Camera variables
CAM_TYPE = "pi_cam"                     # Camera type: web_cam|donkey_gym|pi_cam
IMAGE_RESOLUTION = (IMAGE_H, IMAGE_W, IMAGE_D) = (120, 160, 3)
# Joystick
CONTROLLER_TYPE = "web_ctr"              # web_ctr|ps3
JOYSTICK_THROTTLE_DIR = -1.0,
JOYSTICK_MAX_THROTTLE = 1.0,
JOYSTICK_STEERING_SCALE = 1.0,
AUTO_RECORD_ON_THROTTLE = False,
JOYSTICK_DEADZONE = 0.0
# Train variables
IMG_KEY = "cam/image_array"
STEERING_KEY = "steering"
THROTTLE_KEY = "throttle"
SENSOR_KEYS = ["sensor_fr", "sensor_mr", "sensor_br", "sensor_bl", "sensor_ml", "sensor_fl"]
# SENSOR_KEYS = ["sensor_fr", "sensor_mr", "sensor_br", "sensor_bm", "sensor_bl", "sensor_ml", "sensor_fl"]
# Ultrasonic sensors trigger and echo pins in gpio board mode
# SENSOR_PINS = [(38, 40), (32, 36), (24, 26), (18, 22), (10, 16), (37, 35), (31, 33)]
SENSOR_PINS = [(38, 40), (32, 36), (24, 26), (10, 16), (37, 35), (31, 33)]
SENSOR_NUM = len(SENSOR_KEYS)
TRIG_TIME = 100e-6
SENSOR_TIMEOUT = 0.3    # seconds
MAX_DISTANCE = 2.0      # meters
# Classifier variables
threshold_confidence = 0.8
# Other variables
predict_throttle = False
using_sensors = True
sequence_length = 1
