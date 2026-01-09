import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import os
import pandas as pd

# -------------------------------
# Download model and threshold from Google Drive if not present
# -------------------------------
@st.cache_resource
def download_file(file_id, output_name):
    if not os.path.exists(output_name):
        import gdown  # import inside function for Streamlit Cloud
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_name, quiet=False)
    return output_name

@st.cache_resource
def load_model():
    model_file = download_file("11I5rUQMSpDcFolhzkfvrBwoFYDCyqKtJ", "breast_cancer_efficientnetb3_final.keras")
    model = tf.keras.models.load_model(model_file)
    return model

@st.cache_resource
def load_threshold():
    threshold_file = download_file("YOUR_THRESHOLD_DRIVE_ID", "threshold.pkl")  # replace with Drive ID if threshold on Drive
    with open(threshold_file, "rb") as f:
        threshold = pickle.load(f)
    return threshold

# -------------------------------
# Load resources
# -------------------------------
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
st.markdown("""
Upload breast image(s) (histopathology or mammography).  
The model predicts **Cancer (1)** or **Non-Cancer (0)**.
""")

uploaded_files = st.file_uploader(
    "Choose image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# -------------------------------
# Prediction functions
# -------------------------------
def preprocess_image(image, target_size=(224,224)):
    img = image.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image):
    img_array = preprocess_image(image)
    prob = model.predict(img_array, verbose=0)[0][0]
    prediction = "Cancer" if prob >= threshold else "Non-Cancer"
    return prediction, prob

# -------------------------------
# Run predictions
# -------------------------------
if uploaded_files:
    results = []
    st.write(f"Detected {len(uploaded_files)} file(s)")

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_column_width=True)
        pred, prob = predict(image)
        st.success(f"Prediction: **{pred}** (probability: {prob:.3f})")
        results.append({
            "filename": uploaded_file.name,
            "prediction": pred,
            "probability": prob
        })

    # Display batch table
    df = pd.DataFrame(results)
    st.write("### Batch Predictions")
    st.dataframe(df)

    # CSV download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )
