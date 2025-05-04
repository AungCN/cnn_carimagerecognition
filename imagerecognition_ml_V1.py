import streamlit as st
import numpy as np
import tensorflow as tf
import os
import gdown
from tensorflow.keras.preprocessing import image
from PIL import Image

# Download model if not exists
model_path = "car_classification_model.h5"
if not os.path.exists(model_path):
    # Use direct download URL (replace with your real FILE ID)
    url = "https://drive.google.com/uc?id=1QN8gyLqo4A2-lZwHlbsETyJcfQusMTS3"
    gdown.download(url, model_path, quiet=False)

# Set Streamlit app configuration
st.set_page_config(page_title="Car Classifier", layout="centered")
st.title("🚗 Car Classifier")
st.write("Upload a car image to classify it using the trained CNN model.")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# Class labels
class_labels = [
    'BackInterior', 'BackSide', 'Dashboard', 'FrontInterior', 
    'FrontSide', 'LeftFront', 'LeftMiddle', 'LeftRear', 'RightFront',
    'RightMiddle', 'RightRear', 'Roof', 'Trunk_Keys_Items'
]

# Image uploader
uploaded_file = st.file_uploader("Upload a car image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    predicted_label = class_labels[predicted_index]
    confidence = float(np.max(prediction[0])) * 100

    st.markdown("### Prediction")
    st.write(f"**Predicted Class:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")
