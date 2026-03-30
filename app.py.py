import os
import requests
import streamlit as st
from fastai.vision.all import *

# 1. Page Configuration
st.set_page_config(page_title="AI vs Real Detector", page_icon="🔍")

# 2. Hugging Face Direct Link (Copy panna link-ah inga podunga)
HF_MODEL_URL = "https://huggingface.co/spaces/kuloth/AI-Image-Detector/resolve/main/final_model.pkl"
MODEL_PATH = "final_model.pkl"

@st.cache_resource
def load_my_model():
    # Model download logic
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI Model... Please wait."):
            response = requests.get(HF_MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    
    # Load the learner
    return load_learner(MODEL_PATH)

# Try loading the model
try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error: Model load panna mudiyala. Check link: {e}")
    model = None

# 3. Main UI
st.title("AI vs Real Image Detector 🔍")
st.write("Upload an image to check if it's AI-generated or Real.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    img = PILImage.create(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    with st.spinner('Analyzing...'):
        pred, pred_idx, probs = model.predict(img)
    
    st.divider()
    st.subheader(f"Result: {pred.upper()}")
    st.write(f"Confidence: {probs[pred_idx]*100:.2f}%")
