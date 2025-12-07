import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import io
import os
import tempfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt


# ====================================================
# PAGE CONFIG
# ====================================================
st.set_page_config(
    page_title="Klasifikasi Kaktus",
    page_icon="🌵",
    layout="centered"
)


# ====================================================
# THEME CSS — GREEN CACTUS + AUTO DARK MODE
# ====================================================
st.markdown("""
<style>

:root {
    --cactus-green: #27ae60;
    --cactus-green-dark: #1e874b;
    --soft-green: #e9f7ef;
    --dark-bg: #1c1c1c;
    --dark-card: #242424;
    --text-dark: #e6e6e6;
    --text-light: #333;
}

/* Light mode */
body, .stApp {
    background-color: var(--soft-green);
    color: var(--text-light);
    font-family: 'Segoe UI', sans-serif;
}

/* Content box */
.box {
    padding: 25px;
    border-radius: 14px;
    background: white;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.07);
    transition: 0.3s ease;
}

/* Hover */
.box:hover {
    box-shadow: 0px 6px 25px rgba(0,0,0,0.12);
}

/* Titles */
h1, h2, h3 {
    color: var(--cactus-green-dark);
    font-weight: bold;
}

/* Dark Mode Auto */
@media (prefers-color-scheme: dark) {
    body, .stApp {
        background-color: var(--dark-bg) !important;
        color: var(--text-dark) !important;
    }

    .box {
        background: var(--dark-card) !important;
        color: var(--text-dark) !important;
    }

    h1, h2, h3 {
        color: var(--cactus-green) !important;
    }
}

/* Button */
.stButton>button {
    background: var(--cactus-green) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    font-size: 16px !important;
    border: none !important;
}
.stButton>button:hover {
    background: var(--cactus-green-dark) !important;
}

</style>
""", unsafe_allow_html=True)


# ====================================================
# LOAD TFLITE MODEL
# ====================================================
@st.cache_resource
def load_tflite_model(model_path="model_kaktus.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return (
        interpreter,
        interpreter.get_input_details(),
        interpreter.get_output_details()
    )


# ====================================================
# LOAD CLASS LABELS
# ====================================================
def load_class_labels(path="class_labels.json"):
    if os.path.exists(path):
        return json.load(open(path))
    return ["Astrophytum asteria", "Ferocactus", "Gymnocalycium"]


# ====================================================
# PREPROCESS & PREDICT
# ====================================================
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((150,150))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict(interpreter, input_details, output_details, array):
    interpreter.set_tensor(input_details[0]["index"], array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    probs = np.squeeze(output)
    return probs


# ====================================================
# GENERATE PDF
# ====================================================
def generate_pdf(image, pred_label, probs, labels):

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = temp.name

    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(w/2, h - 60, "Hasil Prediksi Klasifikasi Kaktus")

    # Uploaded image (center & small)
    img_for_pdf = image.copy().convert("RGB")
    img_reader = ImageReader(img_for_pdf)
    img_w = 200
    img_h = 200
    c.drawImage(img_reader, (w-img_w)/2, h - 100 - img_h, img_w, img_h)

    # Prediction text
    y = h - 100 - img_h - 30
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(w/2, y, f"Prediksi : {pred_label}")
    y -= 30

    # Probabilities text
    c.setFont("Helvetica", 12)
    for i, p in enumerate(probs):
        c.drawCentredString(w/2, y, f"{labels[i]} : {p:.4f}")
        y -= 18

    # Bar chart image
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(labels, probs, color=['#2ecc71','#f39c12','#3498db'])
    ax.set_ylim(0,1)
    ax.set_ylabel("Probabilitas")
    ax.set_title("Probabilitas per Kelas")
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)

    chart_reader = ImageReader(buf)
    chart_w = 350
    chart_h = 220
    c.drawImage(chart_reader, (w-chart_w)/2, y - chart_h - 10, chart_w, chart_h)

    c.save()
    buf.close()
    return pdf_path


# ====================================================
# MENU UTAMA (NO SIDEBAR)
# ====================================================
menu = st.selectbox("Menu:", ["🏠 Tentang Kaktus", "🌵 Prediksi Kaktus"])

labels = load_class_labels()


# ====================================================
# PAGE 1 — ABOUT CACTUS
# ====================================================
if menu == "🏠 Tentang Kaktus":

    st.markdown("<h1 style='text-align:center;'>🌵 Tentang Tanaman Kaktus</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class="box">
        <h3>🌱 Sejarah Singkat</h3>
        <p>
            Kaktus berasal dari benua Amerika dan banyak ditemukan di wilayah gurun 
            seperti Meksiko, Arizona, dan Peru. Tanaman ini mampu menyimpan air 
            dalam jumlah besar sehingga dapat bertahan di kondisi lingkungan yang kering.
        </p>

        <h3>✨ Fakta Menarik</h3>
        <ul>
            <li>Kaktus memiliki stomata yang hanya membuka malam hari untuk mengurangi penguapan.</li>
            <li>Banyak kaktus dapat hidup lebih dari 100 tahun.</li>
            <li>Bentuknya beragam: bulat, silinder, hingga bercabang.</li>
            <li>Duri kaktus adalah modifikasi daun untuk melindungi diri.</li>
        </ul>

        <p style="margin-top:15px;">
            Aplikasi ini dibuat untuk membantu mengidentifikasi jenis kaktus 
            menggunakan model Convolutional Neural Network (CNN) yang ringan dan cepat.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ====================================================
# PAGE 2 — PREDIKSI
# ====================================================
elif menu == "🌵 Prediksi Kaktus":

    st.markdown("<h1 style='text-align:center;'>🌵 Prediksi Jenis Kaktus</h1>", unsafe_allow_html=True)

    with st.spinner("Memuat model..."):
        interpreter, input_details, output_details = load_tflite_model()

    uploaded = st.file_uploader("Upload gambar kaktus (jpg/png)", type=["jpg","png","jpeg"])

    if uploaded:
        image = Image.open(uploaded)

        st.markdown("<div class='box' style='text-align:center;'>", unsafe_allow_html=True)
        st.image(image, caption="Gambar yang diupload", width=300)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔍 Prediksi"):
            arr = preprocess(image)
            probs = predict(interpreter, input_details, output_details, arr)

            idx = int(np.argmax(probs))
            pred_label = labels[idx]
            prob = float(probs[idx])

            st.success(f"**Prediksi: {pred_label}** ({prob:.4f})")

            # Barchart
            st.subheader("📊 Grafik Probabilitas")
            fig, ax = plt.subplots()
            ax.bar(labels, probs)
            ax.set_ylabel("Probabilitas")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

            # Table
            st.subheader("📋 Tabel Probabilitas")
            st.table({"Kelas": labels, "Probabilitas": [float(p) for p in probs]})

            # PDF
            pdf_path = generate_pdf(image, pred_label, probs, labels)

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📄 Download PDF Prediksi",
                    data=f,
                    file_name="hasil_prediksi_kaktus.pdf",
                    mime="application/pdf"
                )

    else:
        st.info("Silakan upload gambar untuk memulai prediksi.")
