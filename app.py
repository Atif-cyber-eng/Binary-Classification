# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import io

st.set_page_config(page_title="Fruit Classifier", layout="centered")

st.title("Apple vs Mango — Binary Classifier")
st.write("Upload an image and the model will predict whether it's an apple or a mango.")

# Path to model. When deploying, upload model to the same repo or use an absolute path.
MODEL_PATH = "Fruits_model.h5"  # for Streamlit sharing, put this file in the app repo

@st.cache_resource
def load_tf_model(path):
    model = load_model(path)
    return model

# try load model (provide helpful error if missing)
try:
    model = load_tf_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model from {MODEL_PATH}. Make sure the .h5 file is present. Error: {e}")
    st.stop()

uploaded = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded is not None:
    # display image
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # preprocess
    target_size = (150, 150)
    img_resized = img.resize(target_size)
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)

    # predict
    prob = model.predict(x)[0][0]  # sigmoid output
    label = "mango" if prob >= 0.5 else "apple"
    confidence = prob if prob >= 0.5 else 1 - prob

    st.markdown(f"### Prediction: **{label.upper()}**")
    st.write(f"Confidence: {confidence*100:.2f}%")

    # optional: show raw probability
    st.write(f"Raw model output (sigmoid): {prob:.4f}")
else:
    st.info("Upload an image of an apple or a mango (front-facing photo works best).")
