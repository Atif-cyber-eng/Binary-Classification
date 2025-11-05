import streamlit as st
import numpy as np
import tensorflow as tf # Added the standard TensorFlow import
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown

# --- Streamlit Page Config ---
st.set_page_config(page_title="Fruit Classifier", layout="centered")

st.title("🍎🥭 Apple vs Mango Classifier")
st.write("Upload an image of a fruit and the model will predict whether it's an **Apple** or a **Mango**.")

# --- Google Drive model download ---
MODEL_PATH = "Fruits_model.h5"
DRIVE_FILE_ID = "1jSzKi-F-GeoSwFx-Ts8MpO6PFFmhCB4A"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# Download only if not present (Crucial for remote deployment)
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        # gdown will download the file and name it MODEL_PATH
        try:
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Could not download model from Google Drive. Please check the DRIVE_FILE_ID. Error: {e}")
            st.stop()

# --- Load model ---
# Use st.cache_resource to load the heavy model only once
@st.cache_resource
def load_tf_model(path):
    # This function loads the Keras model
    model = load_model(path)
    return model

try:
    model = load_tf_model(MODEL_PATH)
    st.sidebar.success("Model loaded and ready for prediction!")
except Exception as e:
    st.error(f"❌ Could not load the TensorFlow model. Check dependencies and file integrity. Error: {e}")
    st.stop()

# --- Upload image ---
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess for model
    target_size = (150, 150)
    img_resized = img.resize(target_size)
    # Convert image to numpy array, normalize, and expand dimensions for model input
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    with st.spinner("Predicting..."):
        # The model is trained for binary classification, resulting in a probability score
        prob = model.predict(x)[0][0]

    # Determine label based on probability threshold (0.5)
    if prob >= 0.5:
        label = "Mango 🥭"
        confidence = prob
    else:
        label = "Apple 🍎"
        confidence = 1 - prob

    # Display results
    st.markdown("---")
    st.subheader(f"Prediction: **{label}**")
    st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
    st.markdown("---")
else:
    st.info("👆 Upload an image of an apple or a mango to start classification.")
