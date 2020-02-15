import os
from donkeycar import Vehicle
import tensorflow.compat.v1 as tf_v1
from donkeycar.parts.dgym import DonkeyGymEnv
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController

from src.pid_controller import PIDController
from src.utils import ContextManagerWrapper, parse_args
from src.parts import DonkeyNetController, DriveSelector

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == "__main__":
    args = parse_args(mode="drive")
    GPU_OPTIONS = tf_v1.GPUOptions(per_process_gpu_memory_fraction=0.75)
    TF_CONFIG = tf_v1.ConfigProto(gpu_options=GPU_OPTIONS, allow_soft_placement=True)
    with tf_v1.Session(config=TF_CONFIG) as sess:
        with ContextManagerWrapper(Vehicle(), exit_method="stop") as car:
            SIM_PATH = r"/home/rajk/Machine_Learning/DonkeySimLinux/donkey_sim.x86_64"
            donkey_env = DonkeyGymEnv(SIM_PATH, port=9090, headless=0)
            car.add(donkey_env,
                    inputs=["steering", "throttle"],
                    outputs=["cam/image_array"],
                    threaded=True)

            print("Web controller available at: localhost:{WEB_CONTROLLER_PORT}")
            car.add(LocalWebController(),
                    inputs=["cam/image_array"],
                    outputs=["user/steering", "user/throttle", "user/mode", "recording"],
                    threaded=True)

            input_shape = donkey_env.env.observation_space.shape
            donkey_net_ctr = DonkeyNetController(sess, input_shape=input_shape, model_path=args.model_path,
                                                 version=args.version, throttle=args.throttle)
            car.add(donkey_net_ctr,
                    inputs=["cam/image_array"],
                    outputs=["random/steering", "random/throttle"],
                    run_condition="drive/auto")

            pid_values = {"P": -5.7e-1, "I": -7e-2, "D": -2.2e-1}
            print(f"PID values: {pid_values}")
            pid_ctr = PIDController(**pid_values, min_value=-1.0, max_value=1.0, store_n_errors=20, target_value=0.0)
            car.add(DriveSelector(pid_ctr),
                    inputs=["user/steering", "user/throttle", "random/steering", "random/throttle", "user/mode"],
                    outputs=["steering", "throttle", "drive/auto"])

            if args.recording_path:
                tub_handler = TubHandler(path=args.recording_path)
                tub_inputs = ["cam/image_array", "user/angle", "user/throttle"]
                tub_input_types = ["image_array", "float", "float"]
                car.add(tub_handler.new_tub_writer(inputs=tub_inputs, types=tub_input_types, user_meta=[]),
                        inputs=tub_inputs,
                        outputs=["tub/num_records"],
                        run_condition="recording")
            car.start(rate_hz=args.sim_rate)
