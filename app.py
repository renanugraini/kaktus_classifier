import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import json
import io
import os

# ============================================================
# KONFIGURASI AWAL
# ============================================================
st.set_page_config(
    page_title="Klasifikasi Kaktus",
    page_icon="🌵",
    layout="centered"
)

# ============================================================
# THEME + CSS MODERN
# ============================================================
st.markdown("""
<style>
/* GLOBAL */
body { background-color: #f5f5f5; }

/* ANIMATION */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade { animation: fadeIn 1s ease-in-out; }

/* TITLE */
.main-title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    margin-top: 10px;
    color: #2F2F2F;
}
.subtext {
    text-align: center;
    font-size: 18px;
    color: #555;
}

/* CARD BOX */
.box {
    background: white;
    padding: 25px;
    margin-top: 20px;
    border-radius: 18px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.07);
    transition: 0.2s;
}
.box:hover {
    box-shadow: 0px 12px 35px rgba(0,0,0,0.1);
}

/* PROGRESS ANIMATION */
.progress-text {
    text-align: center;
    font-size: 16px;
    color: #444;
}

/* DARK MODE */
@media (prefers-color-scheme: dark) {
    body { background-color: #1e1e1e; }
    .main-title { color: white; }
    .subtext { color: #ccc; }
    .box { background: #2b2b2b; box-shadow: 0px 8px 25px rgba(255,255,255,0.05); }
    .progress-text { color: #ddd; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("<div class='main-title fade'>🌵 Klasifikasi Tanaman Kaktus</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext fade'>Upload gambar atau ambil foto langsung, dan AI akan mendeteksi jenisnya</div>", unsafe_allow_html=True)

time.sleep(0.5)

# ============================================================
# LOAD MODEL TFLITE
# ============================================================
@st.cache_resource
def load_tflite():
    interpreter = tf.lite.Interpreter("model_kaktus.tflite")
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

interpreter, input_details, output_details = load_tflite()

# ============================================================
# LOAD LABEL
# ============================================================
def load_labels():
    if os.path.exists("class_labels.json"):
        with open("class_labels.json", "r") as f:
            return json.load(f)
    return ["Astrophytum asteria", "Ferocactus", "Gymnocalycium"]

class_labels = load_labels()

# ============================================================
# IMAGE PREPROCESSING
# ============================================================
def preprocess(img):
    img = img.convert("RGB").resize((150,150))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

# ============================================================
# PREDICT FUNCTION
# ============================================================
def predict(img):
    arr = preprocess(img)
    interpreter.set_tensor(input_details[0]['index'], arr.astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    probs = np.squeeze(output)
    idx = int(np.argmax(probs))
    return class_labels[idx], probs

# ============================================================
# MAIN CONTENT BOX
# ============================================================
st.markdown("<div class='box fade'>", unsafe_allow_html=True)

upload_mode = st.radio(
    "Pilih mode input:",
    ["Upload Gambar", "Kamera"],
    horizontal=True
)

img = None

# ============================================================
# MODE UPLOAD NORMAL
# ============================================================
if upload_mode == "Upload Gambar":
    uploaded = st.file_uploader("📤 Upload gambar kaktus", type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Gambar diunggah", use_column_width=True)

# ============================================================
# MODE KAMERA
# ============================================================
else:
    camera_photo = st.camera_input("📷 Ambil Foto")
    if camera_photo:
        img = Image.open(camera_photo)
        st.image(img, caption="Foto berhasil diambil", use_column_width=True)

# ============================================================
# PREDIKSI BUTTON
# ============================================================
if img and st.button("🔍 Prediksi", use_container_width=True):

    # PROGRESS BAR ANIMASI
    progress = st.progress(0)
    label = st.empty()

    for i in range(100):
        time.sleep(0.01)
        progress.progress(i+1)
        label.markdown(f"<div class='progress-text'>Menganalisis... {i+1}%</div>", unsafe_allow_html=True)

    label.empty()

    pred_label, probs = predict(img)

    st.success(f"🌟 Hasil Prediksi: **{pred_label}**")

    st.write("### Probabilitas:")
    for i, p in enumerate(probs):
        st.write(f"- **{class_labels[i]}** → `{p:.4f}`")

st.markdown("</div>", unsafe_allow_html=True)
