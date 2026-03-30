import os
import requests  # Ithu romba mukkiyam!
import streamlit as st
from fastai.vision.all import *

st.set_page_config(page_title="AI vs Real Detector", page_icon="🔍")

# Unga correct-ana resolve link
HF_LINK = "https://huggingface.co/spaces/kuloth/AI-Image-Detector/resolve/main/final_model.pkl"
MODEL_NAME = "final_model.pkl"

@st.cache_resource
def load_my_model():
    # 1. Check for corrupted files
    if os.path.exists(MODEL_NAME) and os.path.getsize(MODEL_NAME) < 1000000:
        os.remove(MODEL_NAME)
        
    # 2. Download from Hugging Face
    if not os.path.exists(MODEL_NAME):
        with st.spinner("Downloading AI Model (89MB)... Idhu oru nimisham aagum."):
            r = requests.get(HF_LINK, allow_redirects=True)
            with open(MODEL_NAME, 'wb') as f:
                f.write(r.content)
    
    return load_learner(MODEL_NAME)

# Main App Logic
try:
    model = load_my_model()
    st.title("AI vs Real Image Detector 🔍")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        img = PILImage.create(uploaded_file)
        st.image(img, caption='Uploaded Image', use_container_width=True)
        
        with st.spinner('Analyzing...'):
            pred, pred_idx, probs = model.predict(img)
            
        st.divider()
        st.subheader(f"Result: {pred.upper()}")
        st.write(f"Confidence: {probs[pred_idx]*100:.2f}%")

except Exception as e:
    st.error(f"Error loading model: {e}")
    if "requests" in str(e):
        st.info("Tip: requirements.txt-la 'requests' add panni commit pannunga.")
