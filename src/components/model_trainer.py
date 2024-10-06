import os
import sys
import pickle
from dataclasses import dataclass
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object
from src.components.param_tuning import HyperParameterTuningConfig

@dataclass
class ModelTrainerConfig:
    trained_model_file_path_template = os.path.join("artifacts", "{}_{}.keras")  # Template for model file path

class ModelTrainer:
    def __init__(self, symbol: str):
        self.model_trainer_config = ModelTrainerConfig()
        self.symbol = symbol  # Store the stock symbol

    def load_best_hyperparameters(self, model_type: str):
        """
        Load the best hyperparameters from a pickle file based on the model type.
        """
        try:
            logging.info("Loading best hyperparameters...")
            hyperparameter_file_path = hyperparameter_file_path = HyperParameterTuningConfig.hyperparameter_file_path_template.format(model_type)

            if not os.path.exists(hyperparameter_file_path):
                raise CustomException(f"Hyperparameter file not found at {hyperparameter_file_path}", sys)

            with open(hyperparameter_file_path, "rb") as file:
                hyperparameters = pickle.load(file)
                logging.info("Best hyperparameters loaded successfully.")
            return hyperparameters
            
        except Exception as e:
            logging.error(f"Error loading hyperparameters: {str(e)}")
            raise CustomException(f"Error loading hyperparameters: {str(e)}", sys)

    def build_model(self, model_type: str, input_shape):
        """
        Build and compile a model based on the model type provided.
        Supported models: LSTM, GRU, etc.
        """
        try:
            if model_type == "LSTM":
                logging.info("Loading best hyperparameters for LSTM model training")
                best_hyperparameters = self.load_best_hyperparameters(model_type)

                model = Sequential()
                model.add(LSTM(units=best_hyperparameters['units_1'], return_sequences=True, input_shape=input_shape))
                model.add(LSTM(units=best_hyperparameters['units_2'], return_sequences=False))
                model.add(Dense(best_hyperparameters['dense_units']))
                # model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
                # model.add(LSTM(50, return_sequences=False))
                # model.add(Dense(25))
                model.add(Dense(1))

                model.compile(optimizer=Adam(learning_rate=best_hyperparameters['learning_rate']),
                            loss='mean_squared_error')
                logging.info(f"{model_type} model compiled successfully.")
                return model
            
            # Add other model types (e.g., GRU) here if needed
            else:
                raise CustomException(f"Model type {model_type} not recognized.", sys)
        except Exception as e:
            logging.error(f"Error building model: {str(e)}")
            raise CustomException(f"Error building model: {str(e)}", sys)

    def initiate_model_trainer(self, x_train, y_train, x_test, y_test, model_type="LSTM"):
        """
        Train the selected model (default is LSTM) and evaluate it.
        """
        try:
            logging.info(f"Received training and testing data for {model_type} model.")

            # Reshaping X_train and X_test to 3D arrays for sequential models like LSTM
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            logging.info(f"Training data shape: {x_train.shape}, Testing data shape: {x_test.shape}")

            # Build the selected model
            model = self.build_model(model_type=model_type, input_shape=(x_train.shape[1], 1))

            # Train the model
            logging.info(f"Starting {model_type} model training...")
            model.fit(x_train, y_train, batch_size=1, epochs=1)  # Increase epochs as needed

            # Save the trained model with the symbol in the filename
            trained_model_file_path = self.model_trainer_config.trained_model_file_path_template.format(model_type,self.symbol)
            model.save(trained_model_file_path)
            logging.info(f"{model_type} model saved to {trained_model_file_path}")

            preprocessor_path = os.path.join('artifacts', f'preprocessor_{self.symbol}.pkl')
            preprocessor = load_object(file_path=preprocessor_path)


            # Evaluate the model on the test set
            logging.info(f"Evaluating the {model_type} model.")
            predictions = model.predict(x_test)

            # Extract the 'scaler' from the preprocessor
            scaler = preprocessor.named_transformers_['num_pipeline'].named_steps['scaler']
            predictions = scaler.inverse_transform(predictions)
            # predictions = preprocessor.inverse_transform(predictions)
            rmse = sqrt(mean_squared_error(y_test, predictions))
            print("rmse = ",rmse)

            logging.info(f"Root Mean Squared Error on test data for {model_type}: {rmse}")

            return rmse

        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise CustomException(f"Error during model training: {str(e)}", sys)
