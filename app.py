# app.py
import streamlit as st
import os
import tempfile
import numpy as np
from PIL import Image

# Hanya download jika file model belum ada
MODEL_URL = "https://drive.google.com/uc?export=download&id=1218hjqElw_zDI698YsCeFlykT_ESbFrh"
MODEL_PATH = "model_kaktus.h5"

@st.cache_data(show_spinner=False)
def download_model(url, local_path):
    import requests
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    else:
        return False

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model, please wait..."):
        success = download_model(MODEL_URL, MODEL_PATH)
        if not success:
            st.error("Gagal download model.")
            st.stop()

# Load model setelah berhasil download
import tensorflow as tf
model = tf.keras.models.load_model(MODEL_PATH)

# Daftar nama kelas — ganti sesuai label datasetmu
class_names = ["Astrophytum asteria", "Ferocactus", "Gymnocalycium"]

st.title("🌵 Klasifikasi Jenis Kaktus")
st.write("Upload gambar kaktus untuk prediksi jenisnya.")

uploaded_file = st.file_uploader("Pilih gambar (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((150, 150))  # sesuaikan dengan ukuran input modelmu
    st.image(img, caption="Preview", use_column_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]) * 100)

    st.markdown(f"### Prediksi: **{class_names[idx]}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
