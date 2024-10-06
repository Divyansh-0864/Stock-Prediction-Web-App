import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.param_tuning import HyperParameterTuning

class TrainPipeline:
    def __init__(self):
        logging.info("Training Pipeline initiated")

    def start_data_ingestion(self, symbol):
        """
        Run the data ingestion step to fetch historical stock data.
        """
        try:
            logging.info("Starting data ingestion.")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion(symbol=symbol)
            logging.info(f"Data ingestion completed. Train data: {train_data_path}, Test data: {test_data_path}")
            return train_data_path, test_data_path
        except Exception as e:
            logging.error("Error during data ingestion.", exc_info=True)
            raise CustomException(e, sys)

    def start_data_transformation(self, train_data_path, test_data_path, symbol):
        """
        Run the data transformation step to preprocess the stock data.
        """
        try:
            logging.info("Starting data transformation.")
            data_transformation = DataTransformation()
            x_train, y_train, x_test, y_test,preprocessor_path = data_transformation.initiate_data_transformation(
                train_path=train_data_path,
                test_path=test_data_path,
                symbol=symbol
            )
            logging.info(f"Data transformation completed. Preprocessor saved at: {preprocessor_path}")
            return (x_train, y_train, x_test, y_test,preprocessor_path)
        except Exception as e:
            logging.error("Error during data transformation.", exc_info=True)
            raise CustomException(e, sys)

    def start_hyperparameter_tuning(self, x_train, y_train, model_type="LSTM"):
        """
        Run hyperparameter tuning to find the best hyperparameters.
        """
        try:
            logging.info("Starting hyperparameter tuning.")
            tuner = HyperParameterTuning(model_type=model_type)
            best_hyperparameters = tuner.tune_hyperparameters(x_train=x_train, y_train=y_train)
            logging.info("Hyperparameter tuning completed.")
            return best_hyperparameters
        except Exception as e:
            logging.error("Error during hyperparameter tuning.", exc_info=True)
            raise CustomException(e, sys)

    def start_model_training(self, x_train, y_train, x_test, y_test, model_type="LSTM", stock="AAPL"):
        """
        Train the model using the best hyperparameters.
        """
        try:
            logging.info("Starting model training.")
            model_trainer = ModelTrainer(stock)
            rmse = model_trainer.initiate_model_trainer(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                model_type=model_type  # Allow specification of model type
            )
            logging.info(f"Model training completed with RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error("Error during model training.", exc_info=True)
            raise CustomException(e, sys)

    def run_training_pipeline(self, symbol, model_type):
        """
        Complete training pipeline to:
        1. Fetch data
        2. Preprocess data
        3. Tune hyperparameters
        4. Train and save the model
        """
        try:
            logging.info("Starting the training pipeline.")

            # Step 1: Data Ingestion
            train_data_path, test_data_path = self.start_data_ingestion(symbol=symbol)

            # Step 2: Data Transformation
            x_train, y_train, x_test, y_test,preprocessor_path = self.start_data_transformation(train_data_path=train_data_path,test_data_path=test_data_path, symbol=symbol)
            
            # Step 3: Hyperparameter Tuning
            best_hyperparameters = self.start_hyperparameter_tuning(x_train, y_train, model_type="LSTM")

            # Step 4: Model Training
            rmse = self.start_model_training(x_train, y_train, x_test, y_test, model_type="LSTM",stock=symbol)  # Default to LSTM

            logging.info("Training pipeline completed successfully.")
            return rmse  # Return RMSE for further analysis

        except Exception as e:
            logging.error("Error in the training pipeline.", exc_info=True)
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        symbol = 'AAPL'  
        model = "LSTM"
        pipeline = TrainPipeline()
        pipeline.run_training_pipeline(symbol, model)
    except Exception as e:
        logging.error("Failed to run the training pipeline.", exc_info=True)
