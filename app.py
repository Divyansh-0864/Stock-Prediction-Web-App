from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline  
import os
from src.logger import logging

app = Flask(__name__)

# Directory to store pre-trained models
PRETRAINED_MODEL_DIR = 'artifacts'


# Route for the home page
@app.route('/')
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

        # if os.path.exists(model_file_path):
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)  # for deployment
    # app.run(debug=True, port=5000)  
