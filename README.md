# Donkey Car - Imitation Learning  
Implementation of Imitation Learning to train a model to drive a donkey car.  
# Requirements  
**OS:** Ubuntu 18.04 (Tested)  
__See `requirements.txt` for information on python libraries.__  

# Train a model
To train a model, go to the main directory of this repository aand execute the command below:  
`python main_train.py --epochs 30 --batch_size 32 --version 3 --data_dir {PATH_TO_YOUR_DATA} --model_path {DIRECTORY_TO_SAVE_MODEL}`
  
This will train the model and save the weights in depending on `model_path` (by default `models` directory) as `model.h5`.

__Please look in `parse_args` function in `src/arg_parser.py` for information on additional parsable arguments__  

# Test a model
The trained model can be tested in simulation provided by [Donkey Gym](https://github.com/tawnkramer/gym-donkeycar).  
Run the following command in the terminal:  
`python main_drive.py --model_path {DIRECTORY_OF_TRAINED_MODEL}`  

__Please look in `parse_args` function in `src/arg_parser.py` for information on additional parsable arguments__  
