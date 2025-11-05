import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown

# --- Page Configuration ---
st.set_page_config(
    page_title="🍎🥭 Fruit Classifier",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f9fafb;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    }
    .title {
        text-align: center;
        font-size: 2.5rem !important;
        font-weight: bold;
        color: #333333;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .upload-box {
        background-color: white;
        border-radius: 15px;
        padding: 1rem;
        border: 2px dashed #d1d5db;
        text-align: center;
    }
    .prediction-box {
        background-color: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #888;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("<div class='title'>🍎🥭 Fruit Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image of a fruit and the model will predict whether it's an Apple or a Mango.</div>", unsafe_allow_html=True)

# --- Google Drive Model Download ---
MODEL_PATH = "Fruits_model.h5"
DRIVE_FILE_ID = "1jSzKi-F-GeoSwFx-Ts8MpO6PFFmhCB4A"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔽 Downloading model from Google Drive..."):
        try:
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Could not download model: {e}")
            st.stop()

# --- Load TensorFlow Model (cached) ---
@st.cache_resource
def load_tf_model(path):
    return load_model(path)

try:
    model = load_tf_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# --- Upload Section ---
st.markdown("<div class='upload-box'>📤 Upload a JPG or PNG image below</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Image preview
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess for model
    target_size = (150, 150)
    img_resized = img.resize(target_size)
    x = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    # Prediction
    with st.spinner("🔍 Analyzing image..."):
        prob = model.predict(x)[0][0]

    label = "🥭 Mango" if prob >= 0.5 else "🍎 Apple"
    confidence = prob if prob >= 0.5 else 1 - prob

    # --- Display Result ---
    st.markdown("""
    <div class='prediction-box'>
        <h3>🔮 Prediction Result</h3>
    """, unsafe_allow_html=True)

    st.markdown(f"<h2 style='color:#2563eb'>{label}</h2>", unsafe_allow_html=True)
    st.metric(label="Confidence", value=f"{confidence*100:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👆 Upload an image to start classification.")

# --- Footer ---
st.markdown("<div class='footer'>Built with ❤️ using Streamlit & TensorFlow</div>", unsafe_allow_html=True)
