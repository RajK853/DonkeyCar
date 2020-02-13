# Donkey Car - Imitation Learning  
Implementation of imitation learning to train a model to drive a donkey car.
#Requirements  
**OS:** Ubuntu 16.04 (Tested)  

# Train a model
For simplicity, `src\models.py` contains `DonkeyNetV0` model which loads a predefined CNN architecture located in `src\layers.py` file.  

To train a model, go to the main directory of this repository aand execute the command below:  
`python main.py --data_dir {PATH_TO_YOUR_DATA} --epochs 50 --verbose`  

This will train the model and save the weights in `models` directory as checkpoint named `model.chkpt`.