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

/* ===== Soft Green Background ===== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1c8f4a 0%, #33a165 50%, #4fb67d 100%);
    background-attachment: fixed;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.25) !important;
    backdrop-filter: blur(4px);
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* ===== ALL TEXT COLOR (biar terlihat) ===== */
h1, h2, h3, h4, h5, h6,
p, label, li, strong, b {
    color: #ffffff !important;
}

/* ===== FILE UPLOADER LABEL ===== */
.stFileUploader > label {
    color: #ffffff !important;
    font-weight: bold;
}

/* ===== Kotak “Card” (semi transparan) ===== */
.stCard {
    background: rgba(255,255,255,0.18) !important;
    padding: 20px;
    border-radius: 14px;
    backdrop-filter: blur(6px);
    box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
}

/* ===== Input teks dan selectbox ===== */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    color: #ffffff !important;
}

/* ===== Buttons ===== */
.stButton>button {
    background-color: #2ecc71 !important;
    color: white !important;
    border-radius: 10px;
    font-weight: bold;
    border: 1px solid #27ae60;
}
.stButton>button:hover {
    background-color: #27ae60 !important;
}

/* ===== Buat ul / li terlihat ===== */
ul li {
    color: #ffffff !important;
    font-size: 16px;
}

/* ===== FORCE VISIBILITY UNTUK FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    background: rgba(0,0,0,0.25) !important;
    padding: 15px !important;
    border-radius: 12px !important;
}

/* ===== BACKGROUND GELAP UNTUK H3 YANG DI DALAM CARD ===== */
.stCard h3 {
    background: rgba(0,0,0,0.20) !important;
    padding: 6px 12px !important;
    border-radius: 8px !important;
    display: inline-block;
    color: #ffffff !important;
}

/* ===== FIX TANPA KOTAK DI ICON UPLOAD ===== */

/* Hilangkan fill di elemen-elemen icon */
[data-testid="stFileUploaderDropzone"] svg rect,
[data-testid="stFileUploaderDropzone"] svg path,
[data-testid="stFileUploaderDropzone"] svg polygon,
[data-testid="stFileUploaderDropzone"] svg line,
[data-testid="stFileUploaderDropzone"] svg circle {
    fill: none !important;
}

/* Styling icon upload */
[data-testid="stFileUploaderDropzone"] svg {
    stroke: #000000 !important;
    background: transparent !important;
    width: 40px !important;
    height: 40px !important;
    margin-bottom: 10px !important;
    display: block !important;
    margin-left: auto !important;
    margin-right: auto !important;
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
        Kaktus merupakan tanaman sukulen unik yang terkenal karena kemampuan menyimpan air dan memiliki duri
        sebagai bentuk adaptasi. Karena kemampuan tersebut, kaktus dapat bertahan hidup di lingkungan ekstrem 
        seperti gurun. Selain tangguh, kaktus juga sering dijadikan tanaman hias karena mudah dirawat dan estetik. 
        Tanaman ini termasuk dalam keluarga <i>Cactaceae</i>.
        </p>
        
<h3>Fakta Menarik Kaktus:</h3>    
<ul>
    <li>Kaktus dapat hidup hingga ratusan tahun.</li>
    <li>Beberapa kaktus dapat tumbuh lebih dari 20 meter.</li>
    <li>Terdapat lebih dari 2.000 spesies kaktus di dunia.</li>
    <li>Bentuknya sangat beragam: bulat, pipih, memanjang, hingga bercabang.</li>
</ul>

<h3>Kegunaan:</h3>
<ul> 
    <li>Tanaman hias: sebagai dekorasi rumah, taman, atau kamar tidur karena estetika dan perawatannya mudah.</li>
    <li>Konsumsi & Kesehatan: Buah dan daun muda kaktus (seperti pir berduri) bisa dimakan, kaya serat, vitamin, mineral untuk kesehatan.</li>
    <li>Bisa juga digunakan dalam produk perawatan kulit.</li>
</ul>

<h3>Jenis Kaktus Tanaman Hias:</h3>
<ul>
    <li>Astrophytum Asteria.</li>
    <li>Ferocactus.</li>
    <li>Gymnocalycium.</li>
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

