import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("face_mask_detection_model.keras")

model = load_model()

# ----------------------------
# Load Image Size
# ----------------------------
with open("image_height.pkl", "rb") as f:
    IMAGE_HEIGHT = pickle.load(f)

with open("image_width.pkl", "rb") as f:
    IMAGE_WIDTH = pickle.load(f)

CLASS_NAMES = ["Mask", "Non Mask"]

# ----------------------------
# Title
# ----------------------------
st.title("😷 Face Mask Detection")
st.write("Upload an image to detect whether a person is wearing a face mask.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

col1, col2, col3 = st.columns([1, 2, 1])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with col2:
        st.image(image, caption="Uploaded Image", width=200)

    img = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

    img = np.array(img, dtype=np.float32) / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    probability = float(prediction[0][0])

    if probability >= 0.5:
        predicted_class = CLASS_NAMES[1]
        confidence = probability
    else:
        predicted_class = CLASS_NAMES[0]
        confidence = 1 - probability

    st.markdown("---")

    if predicted_class == "Mask":
        st.success(f"Prediction : {predicted_class}")
    else:
        st.error(f"Prediction : {predicted_class}")

    st.write(f"Confidence : **{confidence*100:.2f}%**")

    st.subheader("Prediction Scores")

    st.write(f"Mask : {(1-probability)*100:.2f}%")
    st.write(f"Non Mask : {probability*100:.2f}%")