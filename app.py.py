import os
import requests
import streamlit as st
from fastai.vision.all import *

# Page Config
st.set_page_config(page_title="AI vs Real Detector", page_icon="🔍")

# Step 1: Model download panna function
def download_model():
    url = "https://drive.google.com/uc?export=download&id=1Z7dWxjXQCAXQSA_suRnAQCG1SgLZ4Nq"
    output = "final_model.pkl"
    if not os.path.exists(output):
        with st.spinner("Downloading model from Google Drive... Please wait."):
            response = requests.get(url)
            with open(output, "wb") as f:
                f.write(response.content)
    return output

# Step 2: Model-ah load pannunga
@st.cache_resource
def load_my_model():
    model_path = download_model()
    return load_learner(model_path)

try:
    model = load_my_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Step 3: UI for Prediction
st.title("AI vs Real Image Detector 🔍")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    img = PILImage.create(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    with st.spinner('Analyzing...'):
        pred, pred_idx, probs = model.predict(img)
    
    st.divider()
    st.subheader(f"Result: {pred.upper()}")
    st.write(f"Confidence: {probs[pred_idx]*100:.2f}%")