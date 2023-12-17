import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model_path = 'sea_animal_classification_model.h5'
model = tf.keras.models.load_model(model_path)

# Directory containing new images to predict
predict_dir = 'images_to_predict'

# Class labels (same order as training labels)
class_labels = ['Dolphin', 'Eel', 'Penguin', 'Seal', 'Sharks']

def predict_image(img_path):
    print(f"Predicting image: {img_path}")
    img = image.load_img(img_path, target_size=(300, 300))  # Resize the image to match model's expected input
    img_array = image.img_to_array(img)  # Convert the image into array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    confidence = np.max(tf.nn.softmax(prediction[0])) * 100  # Apply softmax and convert to percentage

    return predicted_class, confidence

# Function to check if a file is an image
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))

# Predict each image in the directory
for img_file in os.listdir(predict_dir):
    if is_image_file(img_file):
        img_path = os.path.join(predict_dir, img_file)
        predicted_class, confidence = predict_image(img_path)
        print(f"Image: {img_file}, Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%")
