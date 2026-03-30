import os
import requests
import streamlit as st
from fastai.vision.all import *

st.set_page_config(page_title="AI vs Real Detector", page_icon="🔍")

# Step 1: Secure Model download function
def download_model():
    file_id = "1Z7dWxjXQCAXQSA_suRnAQCG1SgLZ4Nq"
    destination = "final_model.pkl"
    
    if not os.path.exists(destination):
        with st.spinner("Downloading model from Google Drive (89MB)... This might take a minute."):
            def get_confirm_token(response):
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        return value
                return None

            def save_response_content(response, destination):
                CHUNK_SIZE = 32768
                with open(destination, "wb") as f:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk: # filter out keep-alive new chunks
                            f.write(chunk)

            URL = "https://docs.google.com/uc?export=download"
            session = requests.Session()
            response = session.get(URL, params={'id': file_id}, stream=True)
            token = get_confirm_token(response)

            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(URL, params=params, stream=True)
            
            save_response_content(response, destination)
    return destination

# Step 2: Load Model
@st.cache_resource
def load_my_model():
    try:
        model_path = download_model()
        return load_learner(model_path)
    except Exception as e:
        st.error(f"Model load panna mudiyala: {e}")
        return None

model = load_my_model()

# Step 3: UI
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
