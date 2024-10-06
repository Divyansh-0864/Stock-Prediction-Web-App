import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    def __init__(self, symbol: str):
        # Save a unique preprocessor for each stock symbol
        self.preprocessor_obj_file_path = os.path.join('artifacts', f"preprocessor_{symbol}.pkl")
        self.sequence_length: int = 60


class DataTransformation:
    def __init__(self, symbol: str):
        self.data_transformation_config = DataTransformationConfig(symbol)
        self.symbol = symbol

    def create_sequences(self, data, target_column_index, sequence_length):
        """
        Create sequences (windowed data) for LSTM training.
        Takes `sequence_length` previous data points to predict the next target.
        """
        sequences = []
        targets = []

        for i in range(sequence_length, len(data)):
            # Append the sequences (features) and the next target value (close price)
            sequences.append(data[i-sequence_length:i, 0])
            targets.append(data[i, target_column_index])

        return np.array(sequences), np.array(targets)

    def initiate_data_transformation(self, train_path, test_path, symbol):
        """
        Initiates the data transformation process by reading train and test data,
        applying scaling, and creating sequences for LSTM model training.
        """
        try:
            # Read the train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data read successfully.")

            target_column_name = "Close"

            # Impute missing values using median
            imputer = SimpleImputer(strategy="median")
            train_df[target_column_name] = imputer.fit_transform(train_df[[target_column_name]])
            test_df[target_column_name] = imputer.transform(test_df[[target_column_name]])

            logging.info("Missing values imputed.")

            # Scale the training and test data
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_scaled = scaler.fit_transform(train_df[[target_column_name]])
            test_scaled = scaler.transform(test_df[[target_column_name]])

            logging.info("Scaling applied to training and testing datasets.")

            # Create sequences for the LSTM model
            sequence_length = self.data_transformation_config.sequence_length
            train_sequences, train_targets = self.create_sequences(train_scaled, 0, sequence_length)
            test_sequences, test_targets = self.create_sequences(test_scaled, 0, sequence_length)

            logging.info("Sequences and targets created for the LSTM model.")
            logging.info(f"Train sequences shape: {train_sequences.shape}, Train targets shape: {train_targets.shape}")
            logging.info(f"Test sequences shape: {test_sequences.shape}, Test targets shape: {test_targets.shape}")

            # Save the scaler object for the specific stock symbol
            scaler_file_path = self.data_transformation_config.preprocessor_obj_file_path
            logging.info(f"Saving the scaler object for {self.symbol} at {scaler_file_path}.")
            save_object(
                file_path=scaler_file_path,
                obj=scaler
            )

            return (
                train_sequences,  # Features for training (sequences)
                train_targets,    # Labels for training (next day 'Close' price)
                test_sequences,   # Features for testing (sequences)
                test_targets,     # Labels for testing (next day 'Close' price)
                scaler_file_path  # Return the path to the scaler object
            )
        except Exception as e:
            logging.error(f"Error during data transformation for {self.symbol}.")
            raise CustomException(e, sys)
