import subprocess
import sys
import os

# 1. --- AUTOMATED LIBRARY SETUP ---
# Streamlit Cloud-la environment issues-ah bypass panna intha step mukkiyam
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install required packages automatically
try:
    import fastai
    import requests
except ImportError:
    # First time run aagumpothu matum idhu install pannum
    install('fastai==2.7.13')
    install('requests')
    install('ipython')

# 2. --- REGULAR IMPORTS ---
import streamlit as st
from fastai.vision.all import *

# Page Layout Configuration
st.set_page_config(page_title="AI vs Real Detector", page_icon="🔍")

# 3. --- CONFIGURATION & DOWNLOAD LOGIC ---
# Hugging Face Direct Download Link
HF_LINK = "https://huggingface.co/spaces/kuloth/AI-Image-Detector/resolve/main/final_model.pkl"
MODEL_NAME = "final_model.pkl"

@st.cache_resource
def load_my_model():
    # 1. Check for corrupted or half-downloaded files (less than 1MB)
    if os.path.exists(MODEL_NAME) and os.path.getsize(MODEL_NAME) < 1000000:
        os.remove(MODEL_NAME)
        
    # 2. Download from Hugging Face if file doesn't exist
    if not os.path.exists(MODEL_NAME):
        with st.spinner("Downloading AI Model (89MB)... Idhu oru nimisham aagum."):
            try:
                r = requests.get(HF_LINK, allow_redirects=True)
                with open(MODEL_NAME, 'wb') as f:
                    f.write(r.content)
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None
    
    # 3. Load the FastAI Learner
    return load_learner(MODEL_NAME)

# 4. --- MAIN APP UI ---
st.title("AI vs Real Image Detector 🔍")
st.write("Upload a photo to check if it's AI-generated or a Real photo.")

try:
    model = load_my_model()
    
    if model is not None:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # Display uploaded image
            img = PILImage.create(uploaded_file)
            st.image(img, caption='Uploaded Image', use_container_width=True)
            
            # Predict button/logic
            with st.spinner('Analyzing patterns...'):
                pred, pred_idx, probs = model.predict(img)
            
            # Show Results
            st.divider()
            result_color = "red" if pred.lower() == "ai" else "green"
            st.markdown(f"### Result: :{result_color}[{pred.upper()}]")
            st.write(f"**Confidence Level:** {probs[pred_idx]*100:.2f}%")
            
            # Visual progress bar
            st.progress(float(probs[pred_idx]))

except Exception as e:
    st.error(f"Error loading model: {e}")
    # Version issue thirumbavum vandha idhu help pannum
    if "'Resolver' object has no attribute 'dict'" in str(e):
        st.info("Tip: FastAI version issue detected. Requirements.txt-la 'fastai==2.7.13' podunga.")
