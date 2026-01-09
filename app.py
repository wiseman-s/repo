import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import os

# -------------------------------
# Download model from Google Drive if not present
# -------------------------------
@st.cache_resource
def download_model():
    model_path = "breast_cancer_efficientnetb3_final.keras"
    if not os.path.exists(model_path):
        import gdown
        # Google Drive shareable link ID
        file_id = "11I5rUQMSpDcFolhzkfvrBwoFYDCyqKtJ"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    return model_path

@st.cache_resource
def load_model():
    model_path = download_model()
    model = tf.keras.models.load_model(model_path)
    return model

# -------------------------------
# Load threshold
# -------------------------------
@st.cache_resource
def load_threshold():
    threshold_path = "threshold.pkl"
    if not os.path.exists(threshold_path):
        st.error("Threshold file not found!")
        st.stop()
    with open(threshold_path, "rb") as f:
        threshold = pickle.load(f)
    return threshold

model = load_model()
threshold = load_threshold()

# -------------------------------
# App layout
# -------------------------------
st.set_page_config(
    page_title="Breast Cancer Detector",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Breast Cancer Detection")
st.markdown(
    """
Upload a breast image (histopathology / mammography). 
The model predicts **Cancer (1)** or **Non-Cancer (0)**.
"""
)

uploaded_files = st.file_uploader(
    "Choose image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# -------------------------------
# Prediction function
# -------------------------------
def preprocess_image(image, target_size=(224,224)):
    img = image.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image):
    img_array = preprocess_image(image)
    prob = model.predict(img_array)[0][0]
    prediction = "Cancer" if prob >= threshold else "Non-Cancer"
    return prediction, prob

# -------------------------------
# Run predictions
# -------------------------------
if uploaded_files:
    st.write(f"Detected {len(uploaded_files)} file(s)")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_column_width=True)
        pred, prob = predict(image)
        st.success(f"Prediction: **{pred}** (probability: {prob:.3f})")
