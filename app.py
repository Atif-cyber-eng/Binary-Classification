# app.py
import streamlit as st
import numpy as np
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

# Download only if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")

# --- Load model ---
@st.cache_resource
def load_tf_model(path):
    model = load_model(path)
    return model

try:
    model = load_tf_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Could not load model. Error: {e}")
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
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    with st.spinner("Predicting..."):
        prob = model.predict(x)[0][0]

    label = "Mango 🥭" if prob >= 0.5 else "Apple 🍎"
    confidence = prob if prob >= 0.5 else 1 - prob

    st.subheader(f"Prediction: {label}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
else:
    st.info("👆 Upload an image of an apple or a mango to start classification.")
