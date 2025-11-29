import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.title("Klasifikasi Jenis Kaktus")
st.write("Upload gambar kaktus untuk mengetahui jenisnya.")

uploaded_file = st.file_uploader("Upload gambar...", type=["jpg", "png", "jpeg"])

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model_kaktus.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["Astrophytum asteria", "Ferocactus", "Gymnocalycium"]

def predict(image):
    img = image.resize((150,150))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang di-upload")

    result = predict(image)
    idx = np.argmax(result)
    confidence = float(result[idx] * 100)

    st.write(f"### Prediksi: **{class_names[idx]}**")
    st.write(f"### Akurasi: {confidence:.2f}%")

