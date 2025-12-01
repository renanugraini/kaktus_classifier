import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import json
import io
import os

# =========================
# KONFIGURASI AWAL
# =========================
st.set_page_config(
    page_title="Klasifikasi Kaktus",
    page_icon="🌵",
    layout="centered"
)

# =========================
# CSS TEMA MINIMALIS
# =========================
st.markdown("""
<style>
body { background-color: #fafafa; }
.main-title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
    color: #333;
    margin-top: 20px;
}
.subtext {
    text-align: center;
    color: #666;
    font-size: 18px;
}
.box {
    background: white;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER ANIMASI
# =========================
st.markdown("<div class='main-title'>🌵 Klasifikasi Tanaman Kaktus</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Upload gambar dan model akan memprediksi jenis kaktusnya</div>", unsafe_allow_html=True)

with st.spinner("Memuat komponen..."):
    time.sleep(0.8)

st.write("")

# =========================
# LOAD MODEL TFLITE
# =========================
@st.cache_resource
def load_tflite():
    interpreter = tf.lite.Interpreter("model_kaktus.tflite")
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

interpreter, input_details, output_details = load_tflite()


# =========================
# LOAD LABEL KELAS
# =========================
def load_labels():
    if os.path.exists("class_labels.json"):
        with open("class_labels.json", "r") as f:
            return json.load(f)
    return ["Astrophytum asteria", "Ferocactus", "Gymnocalycium"]

class_labels = load_labels()


# =========================
# PREPROCESS GAMBAR
# =========================
def preprocess(img):
    img = img.convert("RGB").resize((150,150))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


# =========================
# PREDIKSI
# =========================
def predict(img):
    arr = preprocess(img)

    interpreter.set_tensor(input_details[0]['index'], arr.astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    probs = np.squeeze(output)

    idx = int(np.argmax(probs))
    return class_labels[idx], probs


# =========================
# UI UPLOAD GAMBAR
# =========================
st.markdown("<div class='box'>", unsafe_allow_html=True)

uploaded = st.file_uploader("📤 Upload gambar kaktus", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Gambar diunggah", use_column_width=True)

    if st.button("🔍 Prediksi"):
        with st.spinner("Menganalisis gambar..."):
            time.sleep(1)
            pred_label, probs = predict(img)

        st.success(f"🌟 Hasil Prediksi: **{pred_label}**")

        # Tampilkan tabel probabilitas
        st.write("### Probabilitas:")
        for i, p in enumerate(probs):
            st.write(f"- **{class_labels[i]}** → `{p:.4f}`")

st.markdown("</div>", unsafe_allow_html=True)
