from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline  
import os
from src.logger import logging

application = Flask(__name__)
app = application

# Directory to store pre-trained models
PRETRAINED_MODEL_DIR = 'artifacts'


# Route for the home page
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Get ticker and model_name from form
        ticker = request.form.get('ticker')
        model_name = request.form.get('model_name', default='LSTM')  # LSTM is default if no model is selected

        # Check if the model for the selected ticker is already trained
        model_file_path = os.path.join(PRETRAINED_MODEL_DIR, f'{model_name}_{ticker}.keras')

        if os.path.exists(model_file_path):
            # Model exists, proceed to prediction
            logging.info(f"Model for {ticker} already exists. Proceeding with prediction.")

            # Initialize the custom data and predict pipeline
            # data = CustomData(ticker=ticker, model_name=model_name)
            # pred_df = data.get_data_as_data_frame()

            # Initialize PredictPipeline
            predict_pipeline = PredictPipeline()

            # Perform prediction
            logging.info("Running prediction...")
            results = predict_pipeline.predict_next_day(symbol=ticker, model_type=model_name)

            logging.info(f"Prediction results: {results}")
            return render_template('home.html', results=results)

        else:
            # Model does not exist, train the model first
            logging.warning(f"Model for {ticker} does not exist. Starting training process...")

            # Initialize training pipeline and notify user of the training
            train_pipeline = TrainPipeline()
            try:
                train_pipeline.run_training_pipeline(ticker, model_name)  # Training the model for the ticker

                # Once training is complete, proceed with prediction
                logging.info(f"Model for {ticker} is trained. Proceeding with prediction.")

                # Initialize PredictPipeline
                predict_pipeline = PredictPipeline()

                # Perform prediction
                logging.info("Running prediction after training...")
                results = predict_pipeline.predict_next_day(symbol=ticker, model_type=model_name)

                logging.info(f"Prediction results after training: {results}")
                return render_template('home.html', results=results[0], message="Model training complete. Prediction complete.")

            except Exception as e:
                logging.error(f"Error during training: {e}", exc_info=True)
                return render_template('home.html', message=f"Error in prediction for {ticker}. Please try again later.")


if __name__ == "__main__":
    # app.run(host="0.0.0.0", debug=True)  # for deployment
    app.run(debug=True, port=5000)  
