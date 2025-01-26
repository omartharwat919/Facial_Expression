import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("emotion_detection_model.h5")

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stSidebar {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stTitle {
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app
st.title("Face Emotion Detection")
st.write("Upload an image to detect emotions.")

# Sidebar for information
st.sidebar.title("About")
st.sidebar.write("This app detects emotions from facial expressions using a deep learning model.")
st.sidebar.write("Created by **Amr Elsherbiny**.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Read and preprocess the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    image = cv2.resize(image, (48, 48))  # Resize to (48, 48)
    image = image.reshape(1, 48, 48, 1) / 255.0  # Normalize and reshape

    # Predict emotion
    prediction = model.predict(image)
    emotion = emotion_labels[np.argmax(prediction)]

    # Display the result
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write(f"Predicted Emotion: **{emotion}**")

# Footer
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px;">
        <p>Developed by <strong>Amr Elsherbiny</strong></p>
        <p>© 2023 All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True,
)