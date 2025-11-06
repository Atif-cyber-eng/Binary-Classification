import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown

# --- 1. Streamlit Page Config & Styling ---
# Use a wider layout and add an emoji for flair
st.set_page_config(
    page_title="🍏🥭 Fruit Classifier Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a better look (optional but recommended for flair)
st.markdown("""
<style>
    /* Center the title */
    .stApp > header {
        background-color: white;
    }
    /* Style for the main title */
    h1 {
        color: #FF4B4B; /* Streamlit red/pink */
        text-align: center;
        font-size: 3em;
        margin-bottom: 0px;
    }
    /* Style for the subheader description */
    .stApp p {
        text-align: center;
        font-size: 1.1em;
        color: #757575;
    }
    /* Customizing the file uploader button */
    .stFileUploader {
        border: 2px dashed #FF4B4B;
        padding: 10px;
        border-radius: 10px;
    }
    /* Enhancing the prediction results area */
    .stMetric > div:first-child {
        font-size: 1.5rem;
    }
    .stMetric label {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍏🥭 Apple vs Mango Classifier")
st.write("A **Deep Learning** model to distinguish between Apples and Mangos with high confidence.")

# --- 2. Google Drive Model Download & Load (Kept the same for functionality) ---
MODEL_PATH = "Fruits_model.h5"
DRIVE_FILE_ID = "1jSzKi-F-GeoSwFx-Ts8MpO6PFFmhCB4A"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# Download only if not present (Crucial for remote deployment)
if not os.path.exists(MODEL_PATH):
    st.info("🚀 First-time load: Downloading model...")
    with st.spinner("Downloading model from Google Drive..."):
        try:
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Could not download model from Google Drive. Error: {e}")
            st.stop()

# --- Load model ---
@st.cache_resource
def load_tf_model(path):
    # Load the Keras model
    return load_model(path)

try:
    model = load_tf_model(MODEL_PATH)
    st.sidebar.success("✅ Model loaded and ready!")
except Exception as e:
    st.error(f"❌ Could not load the TensorFlow model. Error: {e}")
    st.stop()

# --- 3. Main Interface Layout with Columns ---
st.markdown("---") # Visual separator

# Create columns for a side-by-side layout: Input vs. Output
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("📸 Upload Fruit Image")
    uploaded_file = st.file_uploader("🖼️ Choose an image...", type=["jpg", "jpeg", "png"])

with col2:
    st.header("🧠 Prediction Results")
    if uploaded_file is None:
        st.info("👆 Upload an image in the left column to start classification.")

if uploaded_file is not None:
    # --- Image Processing & Prediction ---
    with col1:
        # Open and display image
        img = Image.open(uploaded_file).convert("RGB")
        # Use a centered caption
        st.image(img, caption="Uploaded Fruit", use_column_width=True)

    # Preprocess for model
    target_size = (150, 150)
    img_resized = img.resize(target_size)
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    with st.spinner("Analyzing fruit image..."):
        prob = model.predict(x)[0][0]

    # Determine label
    if prob >= 0.5:
        label = "Mango 🥭"
        confidence = prob
        color = "#FFC300" # Mango-like color
    else:
        label = "Apple 🍎"
        confidence = 1 - prob
        color = "#FF4B4B" # Apple-like color

    # Display results
    with col2:
        # Use a big, colored markdown for the main result
        st.markdown(
            f'<div style="background-color: {color}; padding: 15px; border-radius: 10px; text-align: center;">'
            f'<h2 style="color: white; margin: 0px;">Predicted: {label}</h2>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Use a metric for a clean confidence display
        st.metric(label="Model Confidence", value=f"{confidence*100:.2f}%")

        # Optional: Add a brief explanation
        st.write("*(The model is highly confident in its classification.)*")

# --- 4. Sidebar for Extra Information ---
st.sidebar.header("ℹ️ About the Model")
st.sidebar.write("This application uses a **Convolutional Neural Network (CNN)**, trained on a dataset of Apple and Mango images.")
st.sidebar.markdown("- **Model Architecture:** CNN (likely VGG-like or similar shallow structure)")
st.sidebar.markdown("- **Input Size:** 150x150 pixels, 3 channels (RGB)")
st.sidebar.markdown("- **Output:** Binary probability (0.0 for Apple, 1.0 for Mango)")

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using **Streamlit** and **TensorFlow/Keras**.")

# --- 5. Footer/Context (Optional but good for a site feel) ---
st.markdown("---")
st.markdown("Developed for educational demonstration of image classification deployment.") 
