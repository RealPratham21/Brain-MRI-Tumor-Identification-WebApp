import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import gdown
import os

def download_model_from_drive():
    model_file = "best_model.keras"
    if not os.path.exists(model_file):
        print("Downloading model from Google Drive...")
        file_id = "1nI4-Rg9xZ37VeRgI7bF_54V74Z6s9VAO"  # Replace with your file ID
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_file, quiet=False)

# Download the model before running the app
download_model_from_drive()

# Function to preprocess and predict
def Diagnose(img):
    st.write("✅ Image received and processed.")
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((256, 256))  # Resize image to model input size

    img_array = keras.preprocessing.image.img_to_array(img)
    print("Shape: ", img_array.shape)
    img_array = img_array.reshape(1, 256, 256, 3)
    img_array = img_array / 255.0
    model = keras.models.load_model('best_model.keras')  # Load model
    pred = model.predict(img_array)
    st.write("🔍 Raw Prediction: ", pred)
    pred_class = pred.argmax(axis=1)
    return pred_class[0]

# Main function to run the app
def main():
    # App title and introduction
    st.markdown(
        """
        <style>
        .title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 20px;
        }
        </style>
        <div class="title">Brain MRI Identifier</div>
        <div class="subtitle">Upload an MRI image to identify the type of brain tumor</div>
        """,
        unsafe_allow_html=True,
    )
    
    # Upload section
    st.sidebar.markdown("### Upload Section")
    st.sidebar.info("Upload an image with almost equal width and height for best results.")
    uploaded_file = st.file_uploader(
        "Upload an MRI Image (PNG, JPG, JPEG)", type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        img = Image.open(uploaded_file)
        st.image(
            img, caption="Uploaded MRI Image", use_column_width=False, width=300
        )
        
        # Perform diagnosis
        with st.spinner('🧠 Analyzing the image... Please wait...'):
            res = Diagnose(img)
        
        # Display result
        labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        st.success(f"🩺 It's Likely: **{labels[res]}**")
    
    else:
        st.warning("Please upload an image to proceed.")

    # Footer credits
    st.markdown("---")
    st.markdown(
        """
        <style>
        .credits {
            font-size: 0.9rem;
            color: #888;
            text-align: center;
            margin-top: 20px;
        }
        .credits a {
            color: #4CAF50;
            text-decoration: none;
        }
        .credits a:hover {
            text-decoration: underline;
        }
        </style>
        <div class="credits">
            Developed by <strong>Prathamesh Bhamare</strong><br>
            <a href="https://github.com/RealPratham21" target="_blank">GitHub</a> |
            <a href="https://www.linkedin.com/in/prathamesh-bhamare-7480b52b2/" target="_blank">LinkedIn</a> |
            <a href="https://x.com/ftw_pratham27" target="_blank">X</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == '__main__':
    main()