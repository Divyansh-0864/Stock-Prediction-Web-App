import os
import sys
import pickle
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from kerastuner import HyperModel, RandomSearch  # Ensure kerastuner is installed
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object 

class LSTMHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(LSTM(units=hp.Int('units_1', min_value=64, max_value=256, step=32), 
                       return_sequences=True, input_shape=(60, 1)))  # Adjust input_shape as per your data
        model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=128, step=32), 
                       return_sequences=False))
        model.add(Dense(units=hp.Int('dense_units', min_value=10, max_value=50, step=10)))
        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='mean_squared_error')
        return model


@dataclass
class HyperParameterTuningConfig:
    hyperparameter_file_path_template = os.path.join("artifacts", "{}_hyperparameters.pkl")  # Template for model-specific params


class HyperParameterTuning:
    def __init__(self, model_type: str):
        self.hyperparameter_config = HyperParameterTuningConfig()
        self.model_type = model_type  

    def tune_hyperparameters(self, x_train, y_train):
        """
        Tune hyperparameters for the specified model type using Random Search.
        """
        try:
            logging.info(f"Starting hyperparameter tuning for {self.model_type}...")

            tuner = RandomSearch(
                LSTMHyperModel(),
                objective='val_loss',
                max_trials=5,  
                executions_per_trial=1,
                directory='artifacts/hyperparameter_tuning',
                project_name=f'{self.model_type}_tuning'
            )

            logging.info("Searching for best hyperparameters...")
            tuner.search(x_train, y_train, epochs=5, validation_split=0.2)  # Adjust epochs as needed

            # Get the best hyperparameters
            best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

            # Log the best hyperparameters
            logging.info(f"Best hyperparameters found for {self.model_type}: {best_hp.values}")

            # Save the best hyperparameters to a pickle file
            hyperparameters = {
                'units_1': best_hp.get('units_1'),
                'units_2': best_hp.get('units_2'),
                'dense_units': best_hp.get('dense_units'),
                'learning_rate': best_hp.get('learning_rate'),
            }

            save_object(
                file_path=self.hyperparameter_config.hyperparameter_file_path_template.format(self.model_type),
                obj=hyperparameters
            )

            logging.info(f"Best hyperparameters saved to {self.hyperparameter_config.hyperparameter_file_path_template.format(self.model_type)}")
            return hyperparameters

        except Exception as e:
            logging.error(f"Error during hyperparameter tuning for {self.model_type}: {str(e)}")
            raise CustomException(f"Error during hyperparameter tuning for {self.model_type}: {str(e)}", sys)
