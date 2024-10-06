import sys
import os
from dataclasses import dataclass
import pickle as pck
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
        # Initialize the data transformation config with the stock symbol
        self.data_transformation_config = DataTransformationConfig(symbol)
        self.symbol = symbol

    def get_data_transformer_object(self):
        """
        This function constructs a preprocessing pipeline for the data transformation process.
        """
        try:
            numerical_columns = ["Close"]

            # Create a numerical pipeline with imputation and scaling
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            logging.info(f"Numerical columns for transformation: {numerical_columns}")

            # Column transformer to apply the numerical pipeline to numerical columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.error("Error in get_data_transformer_object method.")
            raise CustomException(e, sys)

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

            # Obtain the preprocessing object
            logging.info(f"Obtaining preprocessing object for symbol: {symbol}.")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Close"
            target_column_index = 0 

            # Scale the training and test data
            train_scaled = preprocessing_obj.fit(train_df[[target_column_name]])
            train_scaled = preprocessing_obj.transform(train_df[[target_column_name]])

            test_scaled = preprocessing_obj.transform(test_df[[target_column_name]])

            logging.info("Scaling applied to training and testing datasets.")

            # Create sequences for the LSTM model
            sequence_length = self.data_transformation_config.sequence_length
            train_sequences, train_targets = self.create_sequences(train_scaled, target_column_index, sequence_length)
            test_sequences, test_targets = self.create_sequences(test_scaled, target_column_index, sequence_length)

            logging.info("Sequences and targets created for the LSTM model.")
            logging.info(f"Train sequences shape: {train_sequences.shape}, Train targets shape: {train_targets.shape}")
            logging.info(f"Test sequences shape: {test_sequences.shape}, Test targets shape: {test_targets.shape}")

            # Save the preprocessor object for the specific stock symbol
            preprocessor_file_path = self.data_transformation_config.preprocessor_obj_file_path
            logging.info(f"Saving the preprocessor object for {symbol} at {preprocessor_file_path}.")
            save_object(
                file_path=preprocessor_file_path,
                obj=preprocessing_obj
            )

            return (
                train_sequences,  # Features for training (sequences)
                train_targets,    # Labels for training (next day 'Close' price)
                test_sequences,   # Features for testing (sequences)
                test_targets,     # Labels for testing (next day 'Close' price)
                preprocessor_file_path
            )
        except Exception as e:
            logging.error(f"Error during data transformation for {symbol}.")
            raise CustomException(e, sys)
