import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown

model_path = "model.h5"
if not os.path.exists(model_path):
    url = "https://drive.google.com/file/d/1QN8gyLqo4A2-lZwHlbsETyJcfQusMTS3/view?usp=sharing"
    gdown.download(url, model_path, quiet=False)

# Set Streamlit app configuration
st.set_page_config(page_title="Car Classifier", layout="centered")
st.title("🚗 Car Classifier")
st.write("Upload a car image to classify it using the trained CNN model.")

# Load the trained model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('car_classification_model.h5')

model = load_model()

#{'BackInterior': 0, 'BackSide': 1, 'Dashboard': 2, 'FrontInterior': 3, 'FrontSide': 4, 'LeftFront': 5, 'LeftMiddle': 6, 'LeftRear': 7, 'RightFront': 8, 'RightMiddle': 9, 'RightRear': 10, 'Roof': 11, 'Trunk_Keys_Items': 12}
#Number of classes: 13
class_labels = [
    'BackInterior', 'BackSide', 'Dashboard', 'FrontInterior', 
    'FrontSide', 'LeftFront', 'LeftMiddle', 'LeftRear', 'RightFront',
      'RightMiddle', 'RightRear', 'Roof', 'Trunk_Keys_Items'
]

# Image uploader
uploaded_file = st.file_uploader("Upload a car image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    predicted_label = class_labels[predicted_index]
    confidence = float(np.max(prediction[0])) * 100

    # Output
    st.markdown("### Prediction")
    st.write(f"**Predicted Class:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")
