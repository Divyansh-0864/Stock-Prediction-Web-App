import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import requests
import pandas_datareader as pdr
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from datetime import datetime

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.param_tuning import HyperParameterTuning
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    stock_symbol: str = "AAPL"  # Default stock symbol


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def fetch_data_from_stooq(self, symbol):
        """Fetch stock data from Stooq and return as DataFrame."""
        logging.info(f"Fetching data for {symbol} from Stooq.")
        try:
            df = pdr.get_data_stooq(symbol, start='2012-01-01', end=datetime.now())
            df = df.reindex(index=df.index[::-1])
            logging.info(f"Data fetched successfully for {symbol}.")
            return df
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise CustomException(
                f"Error fetching data for {symbol}: {e}", sys)

    def check_existing_data(self, symbol):
        """Check if the CSV already contains data for the specified stock symbol."""
        try:
            if os.path.exists(self.ingestion_config.raw_data_path):
                existing_df = pd.read_csv(self.ingestion_config.raw_data_path)
                if 'Symbol' in existing_df.columns and symbol in existing_df['Symbol'].unique():
                    logging.info(
                        f"Data for {symbol} already exists in the file.")
                    return existing_df, True  # Return the existing data and a flag indicating it exists
                else:
                    logging.info(
                        f"Data for {symbol} does not exist in the file. Fetching new data.")
                    return existing_df, False  # Existing data found but symbol is missing
            else:
                logging.info(
                    "Raw data file does not exist. Fetching new data.")
                return pd.DataFrame(), False  # No existing file, so no data
        except Exception as e:
            logging.error(f"Error checking existing data: {e}")
            raise CustomException(
                f"Error checking existing data for {symbol}: {e}", sys)

    def initiate_data_ingestion(self, symbol):
        logging.info("Entered the data ingestion method.")
        try:
            if symbol:
                self.ingestion_config.stock_symbol = symbol

            # Check if data for the specified symbol already exists
            existing_df, data_exists = self.check_existing_data(
                self.ingestion_config.stock_symbol)

            if not data_exists:
                # Fetch new data from Stooq if not present
                new_data = self.fetch_data_from_stooq(
                    self.ingestion_config.stock_symbol)
                # Add symbol column to identify stock data
                new_data['Symbol'] = self.ingestion_config.stock_symbol

                # Combine with existing data if present, else use the new data directly
                df = pd.concat([existing_df, new_data],
                            ignore_index=True) if not existing_df.empty else new_data

                os.makedirs(os.path.dirname(
                    self.ingestion_config.raw_data_path), exist_ok=True)
                df.to_csv(self.ingestion_config.raw_data_path,
                        index=False, header=True)
                logging.info(
                    f"New data for {self.ingestion_config.stock_symbol} appended to raw data CSV.")
            else:
                df = existing_df  # Use the existing data

            # Filter the DataFrame to include only rows with the specified stock symbol
            filtered_df = df[df['Symbol'] == self.ingestion_config.stock_symbol]

            if filtered_df.empty:
                raise CustomException(
                    f"No data available for symbol {self.ingestion_config.stock_symbol}", sys)

            # Train-test split on the filtered data
            logging.info("Train-test split initiated.")
            train_set, test_set = train_test_split(
                filtered_df, test_size=0.2, random_state=42)

            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path,
                            index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=False, header=True)

            logging.info("Data ingestion completed.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise CustomException(f"Error during data ingestion for {symbol}: {e}", sys)

if __name__ == "__main__":
    # For checking to see data_ingestion working
    # obj = DataIngestion()
    # obj.initiate_data_ingestion()

    symbol = 'AMZN'
    model_type = 'LSTM'

    # Check for data_transformation
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion(symbol=symbol)

    data_transformation = DataTransformation(symbol=symbol)
    # data_transformation.initiate_data_transformation(train_data,test_data)
    x_train, y_train, x_test, y_test, _ = data_transformation.initiate_data_transformation(train_data, test_data, symbol=symbol)

    hyperparameter_tuning = HyperParameterTuning(model_type=model_type)

    # Call tune_hyperparameters with your training data
    best_hyperparams = hyperparameter_tuning.tune_hyperparameters(
        x_train, y_train)

    model_trainer = ModelTrainer(symbol=symbol)

    # Call initiate_model_trainer with your training and testing data
    rmse = model_trainer.initiate_model_trainer(
        x_train, y_train, x_test, y_test, model_type=model_type)
    print("Rmse = ", rmse)
