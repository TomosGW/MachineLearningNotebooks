
import json
import onnxruntime
import numpy as np
import cv2
from azureml.core.model import Model
from keras.applications.densenet import preprocess_input
from keras.preprocessing.image import img_to_array

IMAGE_WIDTH, IMAGE_HEIGHT = 120, 120

def init():
    global model_path
    model_path = Model.get_model_path(model_name = 'caries-filter-onnx')

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = np.array(data, dtype=np.uint16)
        
        ## Process and analyse the image
        image = cv2.cvtColor(data,cv2.COLOR_GRAY2RGB)        
        
        # resize the image to a fixed size, then flatten the image into
        # a list of raw pixel intensities
        dim = (IMAGE_WIDTH, IMAGE_HEIGHT)
        image = cv2.resize(image, dim)   
        image = preprocess_input(image)
        
        data = img_to_array(image)
        data = np.expand_dims(data, axis=0)

        session = onnxruntime.InferenceSession(model_path)
        first_input_name = session.get_inputs()[0].name
        first_output_name = session.get_outputs()[0].name
        result = session.run([first_output_name], {first_input_name: data})
        # NumPy arrays are not JSON serialisable
        result = result[0].tolist()

        return {"result": result}
    except Exception as e:
        result = str(e)
        return {"error": result}