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
    initial_sidebar_state="auto" # Changed to 'auto' for better mobile experience
)

# Custom CSS for a better look (MAJOR IMPROVEMENTS HERE)
st.markdown("""
<style>
    /* Global Background and Text */
    .stApp {
        background-color: #f7f9fc; /* Light, soft background */
    }
    .stApp > header {
        background-color: transparent; /* Make header transparent */
    }
    
    /* Title and Subheader */
    h1 {
        color: #e74c3c; /* A slightly deeper red for impact */
        text-align: center;
        font-size: 3.5em; /* Bigger title */
        margin-bottom: 0.1em;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp p {
        text-align: center;
        font-size: 1.2em;
        color: #5d6d7e; /* Softer, professional text color */
        margin-bottom: 25px;
    }
    
    /* File Uploader - Making it a drop zone */
    .stFileUploader {
        border: 3px dashed #3498db; /* Blue for 'action' */
        background-color: #ecf0f1; /* Light gray background */
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        transition: all 0.3s ease-in-out;
    }
    /* Hover effect for file uploader */
    .stFileUploader:hover {
        border-color: #2980b9;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Main Prediction Result Box - Making it stand out */
    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15); /* Soft, professional shadow */
    }
    .prediction-box h2 {
        font-size: 2.5em;
        margin: 0px;
    }
    
    /* Metric Enhancement */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 10px;
        margin-top: 20px;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #eaf2f8; /* Very light blue sidebar */
    }
    
</style>
""", unsafe_allow_html=True)

st.title("🍏🥭 Apple vs Mango Classifier")
st.write("A **Deep Learning** model to accurately distinguish between Apples and Mangos from an image.")


[Image of a deep learning CNN model architecture]


# --- 2. Google Drive Model Download & Load (Kept the same for functionality) ---
MODEL_PATH = "Fruits_model.h5"
DRIVE_FILE_ID = "1jSzKi-F-GeoSwFx-Ts8MpO6PFFmhCB4A"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# Download only if not present
if not os.path.exists(MODEL_PATH):
    st.info("🚀 First-time load: Downloading model...")
    with st.spinner("Downloading **15MB** model from Google Drive..."):
        try:
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False, fuzzy=True) # Added fuzzy=True for robustness
            st.success("Model downloaded successfully! (Cached for next time)")
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
    st.sidebar.success("✅ **Model loaded** and ready to classify!")
except Exception as e:
    st.error(f"❌ Could not load the TensorFlow model. Please check the file path and dependencies. Error: {e}")
    st.stop()

# --- 3. Main Interface Layout with Columns ---
st.markdown("---")

# Use a container for better grouping of the main action area
with st.container():
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1️⃣ Upload Image")
        # Added a clear instruction placeholder
        uploaded_file = st.file_uploader(
            "Upload an image of an Apple or a Mango",
            type=["jpg", "jpeg", "png"],
            help="Supported file types: JPG, JPEG, PNG. Max size: 200MB"
        )
        
        # New Feature: Add a simple example image for user testing
        st.markdown("**or try an example below:**")
        
        # Simple Example button (requires example images to be in the same folder)
        example_image_path = "example_mango.jpg" # Assume you have this file
        if os.path.exists(example_image_path):
             if st.button("🥭 Use Example Mango", help="Click to load a pre-selected image."):
                uploaded_file = example_image_path # Simulates an upload
                st.info("Example image loaded.")
        else:
            st.markdown("*To enable example image, place `example_mango.jpg` in the app directory.*")

    with col2:
        st.subheader("2️⃣ Classification Result")
        if uploaded_file is None or (isinstance(uploaded_file, str) and not os.path.exists(uploaded_file)):
            # This handles both None (no file) and the case where the example file path is used but doesn't exist
            st.info("👆 Upload an image (or use an example) in the left column to see the magic!")

if uploaded_file is not None and (not isinstance(uploaded_file, str) or os.path.exists(uploaded_file)):
    # --- Image Processing & Prediction ---
    try:
        if isinstance(uploaded_file, str):
            # If it's a path (from example button)
            img = Image.open(uploaded_file).convert("RGB")
        else:
            # If it's a Streamlit UploadedFile object
            img = Image.open(uploaded_file).convert("RGB")
            
        # Display image in a clean block
        with col1:
            st.image(img, caption="Image Ready for Analysis", use_column_width=True, width=300) # Added max width for better scaling

        # Preprocess for model
        target_size = (150, 150)
        img_resized = img.resize(target_size)
        x = np.array(img_resized) / 255.0
        x = np.expand_dims(x, axis=0)

        # Predict
        with st.spinner("🧠 Running Deep Learning prediction..."):
            prob = model.predict(x)[0][0]

        # Determine label
        if prob >= 0.5:
            label_text = "Mango 🥭"
            confidence = prob
            color_code = "#f1c40f" # Brighter yellow
        else:
            label_text = "Apple 🍎"
            confidence = 1 - prob
            color_code = "#e74c3c" # Deep red

        # Display results
        with col2:
            # Use the custom prediction box for the main result
            st.markdown(
                f'<div class="prediction-box" style="background-color: {color_code};">'
                f'<h2 style="color: white;">Prediction: {label_text}</h2>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Show a progress bar to visually represent confidence
            st.markdown("### Confidence Level")
            st.progress(float(confidence))
            
            # Use a metric for a clean confidence display
            st.metric(label="Model Confidence Score", value=f"{confidence*100:.2f}%")

            # Add a dynamic, encouraging message
            if confidence > 0.95:
                st.balloons()
                st.success("🎉 Wow! High confidence result - that's a clear image!")
            elif confidence > 0.8:
                 st.info("Great result! The model is very confident in its prediction.")
            else:
                 st.warning("Prediction might be challenging. Try a clearer or more centered image.")


    except Exception as e:
        st.error(f"An error occurred during image processing or prediction: {e}")

# --- 4. Sidebar for Extra Information (Polished) ---
st.sidebar.markdown("---")
st.sidebar.header("🔬 Model Details")
st.sidebar.write("This application uses a **Convolutional Neural Network (CNN)**, trained to classify between Apples and Mangos.")

st.sidebar.info(f"""
- **Architecture:** Custom CNN (Trained on 2 classes)
- **Input Size:** {target_size[0]}x{target_size[1]} pixels (RGB)
- **Output:** Binary probability (Value near **0** is **Apple**, near **1** is **Mango**)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using **Streamlit** and **TensorFlow/Keras**.")
st.sidebar.markdown("Developer: **[Your Name/Alias Here]**")

# --- 5. Footer/Context (Optional but good for a site feel) ---
st.markdown("---")
st.markdown("_*Note:* The model's performance depends on the quality, angle, and clarity of the uploaded image. It only distinguishes between an Apple and a Mango and will be unsure about other fruits._")
