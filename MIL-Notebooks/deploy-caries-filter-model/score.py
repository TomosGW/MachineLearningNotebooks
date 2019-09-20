import json
import numpy as np
import sys
import keras
from keras.models import load_model

from azureml.core.model import Model

def init():
    global model
    print("Executing init() method...")
    print("Python version: " + str(sys.version))
    print("keras version: " + keras.__version__)

    
    model_root = Model.get_model_path('Clean121')
    model = load_model(model_root)
    print("Exiting init() method...")
    
def run(raw_data):
    print("Executing run() method...")

    #data = np.array(json.loads(raw_data)['data'])
    # make prediction
    #y_hat = np.argmax(model.predict(data), axis=1)
    y_hat = [0,1,2,3]
    return y_hat.tolist()