import os
from donkeycar import Vehicle
import tensorflow.compat.v1 as tf_v1
from donkeycar.parts.camera import PiCamera
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.web_controller import LocalWebController
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

from src.pid_controller import PIDController
from src.utils import ContextManagerWrapper, parse_args
from src.parts import DonkeyNetController, DriveSelector

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Simulation path
SIM_PATH = r"/home/rajk/Machine_Learning/DonkeySimLinux/donkey_sim.x86_64"
# Steering variables
STEERING_CHANNEL = 0
STEERING_LEFT_PWM = 100
STEERING_RIGHT_PWM = 260
# Throttle variables
THROTTLE_CHANNEL_LEFT = 3
THROTTLE_FORWARD_PWM_LEFT = 4000
THROTTLE_STOPPED_PWM_LEFT = 2
THROTTLE_REVERSE_PWM_LEFT = 1

if __name__ == "__main__":
    args = parse_args(mode="drive")
    GPU_OPTIONS = tf_v1.GPUOptions(per_process_gpu_memory_fraction=0.75)
    TF_CONFIG = tf_v1.ConfigProto(gpu_options=GPU_OPTIONS, allow_soft_placement=True)
    with tf_v1.Session(config=TF_CONFIG) as sess:
        with ContextManagerWrapper(Vehicle(), exit_method="stop") as car:
            if args.cam_type == "pi_cam":
                img_resolution = (120, 160)
                cam = PiCamera(resolution=img_resolution, framerate=args.sim_rate)
                car.add(cam,
                        outputs=["cam/image_array"],
                        threaded=True)
                # Steering and throttle controllers
                steering_controller = PCA9685(STEERING_CHANNEL)
                steering = PWMSteering(controller=steering_controller,
                                       left_pulse=STEERING_LEFT_PWM,
                                       right_pulse=STEERING_RIGHT_PWM)
                car.add(steering, inputs=['steering'])

                throttle_controller = PCA9685(THROTTLE_CHANNEL_LEFT)
                throttle = PWMThrottle(controller=throttle_controller,
                                       max_pulse=THROTTLE_FORWARD_PWM_LEFT,
                                       zero_pulse=THROTTLE_STOPPED_PWM_LEFT,
                                       min_pulse=THROTTLE_REVERSE_PWM_LEFT)
                car.add(throttle, inputs=['throttle'])
            elif args.cam_type == "donkey_gym":
                from donkeycar.parts.dgym import DonkeyGymEnv
                cam = DonkeyGymEnv(SIM_PATH, port=9090, headless=0, env_name=args.env_name)
                input_shape = cam.env.observation_space.shape
                car.add(cam,
                        inputs=["steering", "throttle"],
                        outputs=["cam/image_array"],
                        threaded=True)
            else:
                raise TypeError(f"Invalid camera type: {args.camera_type}")

            print("Web controller available at: localhost:{WEB_CONTROLLER_PORT}")
            car.add(LocalWebController(),
                    inputs=["cam/image_array"],
                    outputs=["user/steering", "user/throttle", "user/mode", "recording"],
                    threaded=True)

            donkey_net_ctr = DonkeyNetController(sess, input_shape=input_shape, model_path=args.model_path,
                                                 version=args.version, throttle=args.throttle)
            car.add(donkey_net_ctr,
                    inputs=["cam/image_array"],
                    outputs=["random/steering", "random/throttle"],
                    run_condition="drive/auto")

            # TODO: Remove/refactor PID controller?
            pid_values = {"P": -7e-1, "I": -1e-1, "D": -1e-1}
            print(f"PID values: {pid_values}")
            pid_ctr = PIDController(**pid_values, min_value=-1.0, max_value=1.0, store_n_errors=10, target_value=0.0)
            car.add(DriveSelector(pid_ctr),
                    inputs=["user/steering", "user/throttle", "random/steering", "random/throttle", "user/mode"],
                    outputs=["steering", "throttle", "drive/auto"])

            if args.recording_path:
                tub_handler = TubHandler(path=args.recording_path)
                tub_inputs = ["cam/image_array", "steering", "throttle"]
                tub_input_types = ["image_array", "float", "float"]
                car.add(tub_handler.new_tub_writer(inputs=tub_inputs, types=tub_input_types, user_meta=[]),
                        inputs=tub_inputs,
                        outputs=["tub/num_records"],
                        run_condition="recording")
            car.start(rate_hz=args.sim_rate)
