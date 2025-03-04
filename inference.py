import tensorflow as tf
import numpy as np
import cv2

# Load Model
model = tf.keras.models.load_model('cnn_model.h5')

def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32))
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    print(f"Predicted Class: {class_index}")

# Example Usage
predict_image("test_image.jpg")