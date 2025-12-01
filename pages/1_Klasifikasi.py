import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import io

st.title("🌵 Klasifikasi Jenis Kaktus")
st.write("Upload gambar dan sistem akan memprediksi jenisnya.")

# Load label
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# Load model once
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model_kaktus.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Prediction function
def preprocess(img):
    img = img.convert("RGB").resize((150, 150))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0).astype(np.float32)
    return arr

def predict(img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    arr = preprocess(img)
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probs = np.squeeze(output_data)
    pred = np.argmax(probs)
    return pred, probs

uploaded = st.file_uploader("Upload gambar kaktus", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Gambar di-upload", use_column_width=True)

    if st.button("Prediksi"):
        with st.spinner("Sedang memproses..."):
            idx, probs = predict(img)
            st.success(f"Prediksi: **{class_labels[idx]}**")
            st.write("Probabilitas per kelas:")
            for i, p in enumerate(probs):
                st.write(f"- {class_labels[i]} : `{p:.4f}`")
