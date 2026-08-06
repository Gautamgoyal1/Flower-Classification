import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Flower Classifier", page_icon="\U0001F338", layout="centered")

IMG_SIZE = 180

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("flower_classifier.keras")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

st.title("Flower Image Classifier")
st.write("Upload a flower photo and the CNN model will predict its type.")

uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype("float32")  # model applies MobileNetV2 preprocessing internally
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Classifying..."):
        predictions = model.predict(img_array)
        predicted_class = class_names[int(np.argmax(predictions[0]))]
        confidence = 100 * float(np.max(predictions[0]))

    st.success(f"Prediction: **{predicted_class.capitalize()}**")
    st.write(f"Confidence: {confidence:.2f}%")

    st.subheader("Prediction Probabilities")
    probs = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    st.bar_chart(probs)
else:
    st.info("Upload an image to get started.")
