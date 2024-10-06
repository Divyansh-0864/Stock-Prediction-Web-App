# helper function like reading a file
import os
import sys
from keras.models import load_model
from src.logger import logging

import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_model_and_preprocessor(symbol, model_type):
        try:
            model_path = os.path.join("artifacts", f"{model_type}_{symbol}.keras")
            preprocessor_path = os.path.join('artifacts', f'preprocessor_{symbol}.pkl')
            
            logging.info(f"Loading model and preprocessor for {symbol}.")
            model = load_model(model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            return model, preprocessor
        except Exception as e:
            raise CustomException(f"Error loading model or preprocessor: {str(e)}", sys)