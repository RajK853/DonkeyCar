import os
from donkeycar import Vehicle
import tensorflow.compat.v1 as tf_v1
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

import config
from src.utils import ContextManagerWrapper, parse_args, get_camera
from src.parts import DonkeyNetController, DriveSelector, ThrottleGPIOController, UltraSonic, FakeSensor

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == "__main__":
    args = parse_args(mode="drive")
    GPU_OPTIONS = tf_v1.GPUOptions(per_process_gpu_memory_fraction=0.75)
    TF_CONFIG = tf_v1.ConfigProto(gpu_options=GPU_OPTIONS, allow_soft_placement=True)
    with tf_v1.Session(config=TF_CONFIG) as sess:
        with ContextManagerWrapper(Vehicle(), exit_method="stop") as car:
            if config.CAM_TYPE == "donkey_gym":
                from donkeycar.parts.dgym import DonkeyGymEnv
                cam = DonkeyGymEnv(config.SIM_PATH, port=9090, headless=0, env_name=args.env_name)
                input_shape = cam.env.observation_space.shape
                car.add(cam,
                        inputs=["steering", "throttle"],
                        outputs=["cam/image_array"],
                        threaded=True)
            else:
                input_shape = config.IMAGE_RESOLUTION
                Camera = get_camera(config.CAM_TYPE)
                cam = Camera(image_w=config.IMAGE_W, image_h=config.IMAGE_H, image_d=config.IMAGE_D,
                             framerate=config.SIM_RATE)
                car.add(cam,
                        outputs=["cam/image_array"],
                        threaded=True)
                        
            if config.CONTROLLER_TYPE == "web_ctr":
                print("Web controller available at: localhost:{WEB_CONTROLLER_PORT}")
                ctr = LocalWebController()
            else:
                from donkeycar.parts.controller import get_js_controller
                ctr = get_js_controller(config)
            car.add(ctr,
                    inputs=["cam/image_array"],
                    outputs=["user/steering", "user/throttle", "user/mode", "recording"],
                    threaded=True)

            if config.INCLUDE_SENSORS:
                for sensor_type, pins in zip(config.SENSOR_KEYS, config.SENSOR_PINS):
                    trig_pin, echo_pin = pins
                    # sensor = UltraSonic(trig_pin=trig_pin, echo_pin=echo_pin)
                    sensor = FakeSensor()
                    car.add(sensor,
                            outputs=[sensor_type],
                            run_condition="recording")

            if config.SEQUENCE_LENGTH > 1:
                input_shape = (config.SEQUENCE_LENGTH, ) + input_shape
            donkey_net_ctr = DonkeyNetController(sess, input_shape=input_shape, model_path=args.model_path,
                                                 version=args.version, throttle=args.throttle, config=config)
            inputs = ["cam/image_array"]
            if config.INCLUDE_SENSORS:
                inputs.extend(config.SENSOR_KEYS)
            car.add(donkey_net_ctr,
                    inputs=inputs,
                    outputs=["donkeynet/steering", "donkeynet/throttle"],
                    run_condition="drive/auto")

            car.add(DriveSelector(),
                    inputs=["user/steering", "user/throttle", "donkeynet/steering", "donkeynet/throttle", "user/mode"],
                    outputs=["steering", "throttle", "drive/auto"])

            if config.CAM_TYPE != "donkey_gym":
                # Steering and throttle controllers
                steering_controller = PCA9685(config.STEERING_CHANNEL)
                steering = PWMSteering(controller=steering_controller,
                                       left_pulse=config.STEERING_LEFT_PWM,
                                       right_pulse=config.STEERING_RIGHT_PWM)
                car.add(steering, inputs=["steering"])

                throttle_gpio_ctr = ThrottleGPIOController(pwm=7, ain1=12, ain2=11, stby=13)
                car.add(throttle_gpio_ctr, inputs=["throttle"], outputs=["gpio_ctr/throttle"])

                throttle_controller = PCA9685(config.THROTTLE_CHANNEL_LEFT)
                throttle = PWMThrottle(controller=throttle_controller,
                                       max_pulse=config.THROTTLE_FORWARD_PWM_LEFT,
                                       zero_pulse=config.THROTTLE_STOPPED_PWM_LEFT,
                                       min_pulse=config.THROTTLE_REVERSE_PWM_LEFT)
                car.add(throttle, inputs=["gpio_ctr/throttle"])

            if args.recording_path:
                tub_handler = TubHandler(path=args.recording_path)
                tub_inputs = ["cam/image_array", "steering", "throttle"]
                tub_input_types = ["image_array", "float", "float"]
                if config.INCLUDE_SENSORS:
                    tub_inputs.extend(config.SENSOR_KEYS)
                    tub_input_types.extend(["float" for _ in range(config.SENSOR_NUM)])
                car.add(tub_handler.new_tub_writer(inputs=tub_inputs, types=tub_input_types, user_meta=[]),
                        inputs=tub_inputs,
                        outputs=["tub/num_records"],
                        run_condition="recording")
            car.start(rate_hz=config.SIM_RATE)
