import os
import numpy as np
from PIL import Image
import cv2
from keras import models
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model
model = models.load_model('BrainTumor10Epoch.h5')  
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    print(classNo)
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

def getResult(img_path, model):
    image = cv2.imread(img_path)

    # Ensure the image is loaded correctly
    if image is None:
        print(f"Error: Unable to load image from path {img_path}")
        return None

    img = Image.fromarray(image, 'RGB')
    img = img.resize((64, 64))  # Resize image to match input size of the model

    img = np.array(img)
    img = img / 255.0  # Normalize image to [0, 1]

    # Expand dimensions to match the input shape of the model (1, 64, 64, 3)
    input_img = np.expand_dims(img, axis=0)

    # Make a prediction
    result = model.predict(input_img)

    # Convert the result to binary (0 or 1) using a threshold
    binary_result = (result[0][0] > 0.5).astype(int)

    print(binary_result)
    return binary_result

# Route to render index.html (assuming it exists in your templates folder)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the POST request
        f = request.files['file']
        
        # Save the file to a secure location
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Get prediction result
        value = getResult(file_path, model)  # Pass the model argument
        result = get_className(value)
        
        # Return the result to the client
        return result

    return None

if __name__ == '__main__':
    app.run(debug=True)
