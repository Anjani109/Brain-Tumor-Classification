import cv2
from keras import models
from PIL import Image
import numpy as np

# Load the model
model = models.load_model('BrainTumor10Epoch.h5')

# Load and preprocess the image
image_path = 'C:\\Users\\ANJANI\\Desktop\\Brain Tumor\\pred\\pred0.jpg'
image = cv2.imread(image_path)

# Ensure the image is loaded correctly
if image is None:
    print(f"Error: Unable to load image from path {image_path}")
else:
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

    # Print the binary result
    print(binary_result)
