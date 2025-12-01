import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
import io, json, os, time
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Klasifikasi Kaktus",
    page_icon="🌵",
    layout="centered"
)

# ============================================================
# CSS + THEME + LOGO
# ============================================================
st.markdown("""
<style>

/* BG Gradient */
body {
    background: linear-gradient(120deg, #f9f9f9, #f1fdf6);
}

/* Fade animation */
@keyframes fadeIn {
  from {opacity: 0; transform: translateY(6px);}
  to {opacity: 1; transform: translateY(0);}
}
.fade { animation: fadeIn 1.2s ease-in-out; }

/* Header */
.title {
  font-size: 42px;
  font-weight: 800;
  text-align: center;
  margin-top: 5px;
  color: #2b2b2b;
}
.subtext {
  text-align: center;
  color: #666;
  font-size: 18px;
}

/* Card */
.card {
  background: white;
  padding: 22px;
  border-radius: 18px;
  box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
  transition: 0.25s;
}
.card:hover {
  transform: translateY(-3px);
  box-shadow: 0px 14px 28px rgba(0,0,0,0.12);
}

/* Button */
button {
  transition: 0.2s;
}
button:hover {
  transform: scale(1.03);
}

/* DARK MODE */
@media (prefers-color-scheme: dark) {
 body { background: #1a1a1a; }
 .title { color: white; }
 .subtext { color: #ccc; }
 .card {
    background: #2b2b2b;
    box-shadow: 0px 6px 20px rgba(255,255,255,0.05);
 }
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("<div class='title fade'>🌵 Klasifikasi Tanaman Kaktus</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext fade'>Upload atau ambil foto — biarkan AI mengenali jenis kaktus</div>", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_tflite():
    interpreter = tf.lite.Interpreter("model_kaktus.tflite")
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

interpreter, input_details, output_details = load_tflite()

# ============================================================
# LABEL
# ============================================================
def load_labels():
    with open("class_labels.json", "r") as f:
        return json.load(f)
        
labels = load_labels()

# ============================================================
# ENHANCEMENT (Auto Improve)
# ============================================================
def auto_enhance(img):
    img = ImageEnhance.Brightness(img).enhance(1.1)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Color(img).enhance(1.1)
    return img

# ============================================================
# PREDICT
# ============================================================
def predict(img):
    img = img.convert("RGB").resize((150,150))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    o = interpreter.get_tensor(output_details[0]['index'])[0]

    idx = int(np.argmax(o))
    return labels[idx], o

# ============================================================
# MAIN CARD
# ============================================================
st.markdown("<div class='card fade'>", unsafe_allow_html=True)

mode = st.radio("Pilih sumber gambar:", ["📤 Upload", "📷 Kamera"], horizontal=True)
img = None

# Mode upload
if mode == "📤 Upload":
    uploaded = st.file_uploader("Upload gambar kaktus", type=["jpg","png","jpeg"])
    if uploaded:
        img = Image.open(uploaded)

# Mode kamera
else:
    cam = st.camera_input("Ambil Foto")
    if cam:
        img = Image.open(cam)

# ============================================================
# Jika gambar ada → tampilkan
# ============================================================
if img:
    st.image(img, caption="Gambar asli", width=300)

    enhanced = auto_enhance(img)
    st.image(enhanced, caption="Gambar setelah peningkatan kualitas", width=300)

    # Tombol prediksi
    if st.button("🔍 Prediksi", use_container_width=True):

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)

        label, probs = predict(enhanced)

        st.success(f"🌟 Jenis Kaktus: **{label}**")
        
        # Probabilities
        st.write("### Probabilitas")
        for i,p in enumerate(probs):
            st.write(f"- **{labels[i]}** → `{p:.4f}`")

        # ============================================================
        #  BAR CHART PROBABILITY
        # ============================================================
        fig, ax = plt.subplots()
        ax.bar(labels, probs)
        ax.set_title("Grafik Probabilitas")
        st.pyplot(fig)

        # ============================================================
        #  DOWNLOAD HASIL
        # ============================================================
        if st.button("⬇️ Download hasil prediksi (.txt)", use_container_width=True):
            text = f"Hasil Prediksi\nJenis: {label}\n\nProbabilitas:\n"
            for i,p in enumerate(probs):
                text += f"- {labels[i]}: {p:.4f}\n"

            st.download_button("Simpan File", text, file_name="hasil_prediksi.txt")

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN INFORMASI MODEL
# ============================================================
with st.expander("📘 Tentang Model"):
    st.write("""
### Arsitektur Model
Model ini menggunakan CNN (Convolutional Neural Network) dengan TensorFlow Lite.

### Dataset
3 jenis kaktus:
- Astrophytum asteria
- Ferocactus
- Gymnocalycium

### Augmentasi:
- Rotation  
- Zoom  
- Horizontal flip  
- Vertical flip  

### Training
- Epochs: 50  
- Input size: 150 × 150 px  
- Batch size: 32  
    """)
