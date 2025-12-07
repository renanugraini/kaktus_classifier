import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Kaktus Classifier",
    page_icon="🌵",
    layout="centered"
)

# =========================================================
# CUSTOM THEME (GREEN CACTUS + DARK OVERLAY)
# =========================================================

page_bg = """
<style>
/* Background image */
[data-testid="stAppViewContainer"] {
    background-image: url('https://i.ibb.co/s23bD7r/cactus-bg-green.jpg');
    background-size: cover;
    background-repeat: no-repeat;
}

/* Dark overlay */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top:0; left:0;
    width: 100%; height: 100%;
    background: rgba(0,0,0,0.45);
    z-index: 0;
}

/* Bring content to front */
.block-container {
    position: relative;
    z-index: 1;
}

/* Card style */
.stCard {
    background: rgba(0,0,0,0.35);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(6px);
}

/* Text color */
h1, h2, h3, p, label, span {
    color: #ecf0f1 !important;
}

/* Buttons */
.st-emotion-cache-7ym5gk, .st-emotion-cache-7ym5gk:hover {
    background-color: #27ae60 !important;
    color: white !important;
}

/* Selectbox */
.stSelectbox, .st-emotion-cache-1gulkj5 {
    color: white !important;
}

/* Fix sidebar agar tidak ikut gelap */
[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.25) !important;
    backdrop-filter: blur(4px);
}

[data-testid="stSidebar"] * {
    color: #ecf0f1 !important;
}

/* TEKS AGAR TERLIHAT */
h1, h2, h3, h4, h5, h6,
p, span, label, li, strong, b {
    color: #ffffff !important;
}

/* Label upload gambar */
.stFileUploader label {
    color: #ffffff !important;
}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =========================================================
# LOAD TFLITE MODEL
# =========================================================
@st.cache_resource
def load_tflite():
    interpreter = tf.lite.Interpreter(model_path="model_kaktus.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# label kelas
labels = ["Cereus", "Epiphyllum", "Opuntia"]

# =========================================================
# FUNCTION PREDIKSI
# =========================================================
def predict(img):
    image = img.resize((150,150))
    arr = np.array(image)/255.0
    arr = np.expand_dims(arr, axis=0).astype("float32")

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    return preds

# =========================================================
# HALAMAN MENU
# =========================================================

menu = st.sidebar.radio("Navigasi", ["Informasi Kaktus", "Prediksi Kaktus"])

# =========================================================
# PAGE 1: INFORMASI KAKTUS
# =========================================================
if menu == "Informasi Kaktus":
    st.markdown("<h1 class='stCard'>🌵 Informasi Tentang Kaktus</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class='stCard'>
        <h3>Apa itu Kaktus?</h3>
        <p>
        Kaktus merupakan tanaman sukulen yang mampu bertahan hidup di lingkungan ekstrem seperti gurun.
        Mereka menyimpan air di batangnya, memiliki duri sebagai bentuk adaptasi, dan termasuk dalam 
        keluarga <i>Cactaceae</i>.
        </p>

        <h3>Fakta Menarik Kaktus:</h3>
<ul>
    <li>Kaktus dapat hidup hingga ratusan tahun.</li>
    <li>Beberapa kaktus dapat tumbuh lebih dari 20 meter.</li>
    <li>Terdapat lebih dari 2.000 spesies kaktus di dunia.</li>
    <li>Bentuknya sangat beragam: bulat, pipih, memanjang, hingga bercabang.</li>
</ul>

<h3>Jenis Kaktus Yang Sering Dijumpai:</h3>
<ul>
    <li><b>Cereus</b> – bentuk panjang menjulang seperti tiang.</li>
    <li><b>Epiphyllum</b> – memiliki daun pipih & bunga besar.</li>
    <li><b>Opuntia</b> – dikenal sebagai “prickly pear”, bentuk oval pipih.</li>
</ul>
""", unsafe_allow_html=True)

# =========================================================
# PAGE 2: PREDIKSI KAKTUS
# =========================================================
elif menu == "Prediksi Kaktus":
    st.markdown("<h1 class='stCard'>🔍 Prediksi Jenis Kaktus</h1>", unsafe_allow_html=True)
    st.write("Upload gambar kaktus untuk diklasifikasikan menggunakan model CNN.")

    uploaded = st.file_uploader("Upload Gambar", type=["jpg","png","jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        st.markdown("<h3 style='text-align:center;'>Gambar yang diupload</h3>", unsafe_allow_html=True)
        st.image(img, width=280, caption="Preview", use_container_width=False)

        preds = predict(img)
        probs = preds / np.sum(preds)
        kelas = labels[np.argmax(probs)]

        st.markdown(
            f"""
            <div class='stCard'>
                <h2>Hasil Prediksi</h2>
                <p><b>Jenis Kaktus:</b> {kelas}</p>
                <p><b>Probabilitas:</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ===== BAR CHART =====
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(labels, probs)
        ax.set_ylim(0,1)
        ax.set_ylabel("Probabilitas")
        ax.set_title("Probabilitas per Kelas")
        st.pyplot(fig)

