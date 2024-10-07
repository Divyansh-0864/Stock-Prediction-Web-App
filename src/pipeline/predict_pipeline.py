import sys
import pandas as pd
import numpy as np
import pandas_datareader as pdr
from src.exception import CustomException
from src.utils import load_object, save_object, load_model_and_preprocessor
import os
from datetime import datetime, timedelta
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

from src.logger import logging
from src.pipeline.train_pipeline import TrainPipeline
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

class PredictPipeline:
    def __init__(self):
        self.pretrained_models = ['AAPL', 'AMZN', 'GOOG', 'MSFT']  # Pre-trained stock models

    def check_model_existence(self, symbol, model_type):
        model_path = os.path.join("artifacts", f"{model_type}_{symbol}.keras")
        return os.path.exists(model_path)

    def dynamically_train_model(self, symbol, model_type):
        try:
            # Initialize and run the training pipeline
            logging.info(f"Training model for {symbol} dynamically.")
            train_pipeline = TrainPipeline()
            train_pipeline.run_training_pipeline(symbol, model_type=model_type)
            logging.info(f"Model for {symbol} trained and saved.")
        except Exception as e:
            raise CustomException(f"Error dynamically training model for {symbol}: {str(e)}", sys)

    def predict_next_day(self, symbol, model_type="LSTM"):
        try:
            logging.info(f"Predicting next day's stock price for {symbol}.")
            
            # Fetch historical data
            custom_data = CustomData(ticker=symbol, model_name=model_type)
            last_60_days_df = custom_data.get_data_as_data_frame()
            
            # Check if the model exists or needs to be trained dynamically
            if symbol not in self.pretrained_models or not self.check_model_existence(symbol, model_type):
                logging.info(f"No pre-trained model found for {symbol}, training dynamically.")
                self.dynamically_train_model(symbol, model_type)

            # Load the model and preprocessor
            model, preprocessor = load_model_and_preprocessor(symbol, model_type)
            scaler = StandardScaler()
            # scaler=preprocessor.named_transformers_['num_pipeline'].named_steps['scaler']
            
            # Preprocess the features (last 60 days) for prediction
            # data_scaled = scaler.transform(last_60_days_df)
            data_scaled = scaler.fit_transform(last_60_days_df)
            
            x_next_day = np.array([data_scaled])  
            x_next_day = np.reshape(x_next_day, (x_next_day.shape[0], x_next_day.shape[1], 1))

            # Predict the next day's price
            next_day_prediction = model.predict(x_next_day)

            # Inverse transform the predicted value to original scale
            next_day_prediction = scaler.inverse_transform(next_day_prediction)
            predicted_price = next_day_prediction[0][0]

            logging.info(f"Predicted Stock Price for {symbol}: {predicted_price}")
            return predicted_price
        
        except Exception as e:
            logging.error(f"Error during next day prediction: {str(e)}")
            raise CustomException(f"Error during next day prediction: {str(e)}", sys)

class CustomData:
    def __init__(self, ticker: str, model_name: str):
        self.ticker = ticker
        self.model_name = model_name

    def get_data_as_data_frame(self):
        try:
            # Fetch last 60 days of stock data
            df = pdr.get_data_stooq(self.ticker, start="2024-01-01", end=datetime.now())

            data = df.filter(['Close'])
            last_60_days_data = data[:61].values

            if data.empty:
                raise CustomException(f"No data fetched for {self.ticker}. Check the ticker symbol.", sys)

            # Return only the 'Close' prices
            last_60_days_df = pd.DataFrame(last_60_days_data,columns=['Close'])
            return last_60_days_df

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    model = "LSTM"
    symbol = "GOOG"
    obj = PredictPipeline()
    price = obj.predict_next_day(symbol=symbol, model_type=model)
    print(price)