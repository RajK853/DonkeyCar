# Donkey Car - Imitation Learning  
Implementation of imitation learning to train a model to drive a donkey car.
#Requirements  
**OS:** Ubuntu 18.04 (Tested)  
__See `requirements.txt` for information on python libraries.__  

# Train a model
For simplicity, `src\models.py` contains `DonkeyNet` model which loads a predefined CNN architecture located in `src\layers.py` file.  

To train a model, go to the main directory of this repository aand execute the command below:  
`python main_train.py --epochs 30 --version 1 --data_dir {PATH_TO_YOUR_DATA} --retrain_model {MODEL_TO_RETRAIN} --save_model_path {DIRECTORY_TO_SAVE_MODEL} --verbose`
  
__Please look in parse_args function in `utils.py` for information on additional parsable arguments__  

This will train the model and save the weights in depending on `save_model_path` (by default `models` directory) as checkpoint named `model.chkpt`.

# Test a model
The trained model can be tested in simulation provided by [Donkey Gym](https://github.com/tawnkramer/gym-donkeycar).  
Run the following command in the terminal:  
`python main_drive.py --version 1 --model_path {DIRECTORY_OF_TRAINED_MODEL}`  

__Please look in parse_args function in `utils.py` for information on additional parsable arguments__  