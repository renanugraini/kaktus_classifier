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
import requests

# ====================================================
# PAGE CONFIG & STYLE
# ====================================================
st.set_page_config(page_title="Klasifikasi Kaktus", page_icon="🌵", layout="centered")

st.markdown("""
<style>
body { background-color: #fafafa; }
.main-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: #333;
    margin-top: 20px;
}
.subtext {
    text-align: center;
    font-size: 18px;
    color: #666;
}
.box {
    background: white;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.06);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🌵 Klasifikasi Tanaman Kaktus</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Aplikasi sederhana untuk memprediksi jenis kaktus</div>", unsafe_allow_html=True)
st.write("")

# ====================================================
# PAGE SELECTION TANPA SIDEBAR
# ====================================================
page = st.radio("Menu Utama:", ["Fakta & Sejarah Kaktus", "Prediksi Kaktus"])

# ====================================================
# LOAD MODEL & LABELS
# ====================================================
@st.cache_resource
def load_tflite_model(model_path="model_kaktus.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

def load_class_labels(path="class_labels.json"):
    if os.path.exists(path):
        return json.load(open(path))
    return ["Astrophytum asteria", "Ferocactus", "Gymnocalycium"]

interpreter, input_details, output_details = load_tflite_model()
labels = load_class_labels()

# ====================================================
# PREPROCESS & PREDICT FUNCTIONS
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
# PDF GENERATION FUNCTION
# ====================================================
def generate_pdf(image, pred_label, probs, labels):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = temp.name
    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(w/2, h - 80, "Hasil Prediksi Klasifikasi Kaktus")

    # Image
    img_for_pdf = image.copy().convert("RGB")
    img_reader = ImageReader(img_for_pdf)
    img_w = 240
    img_h = 240
    c.drawImage(img_reader, (w - img_w)/2, h - 120 - img_h, img_w, img_h)

    # Prediction text
    c.setFont("Helvetica-Bold", 14)
    y_text = h - 120 - img_h - 30
    c.drawCentredString(w/2, y_text, f"Prediksi : {pred_label}")

    # Probabilities table
    c.setFont("Helvetica", 12)
    y_text -= 25
    for i, p in enumerate(probs):
        c.drawCentredString(w/2, y_text, f"{labels[i]} : {p:.4f}")
        y_text -= 18

    # Bar chart
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(14,5))
    ax.bar(labels, probs, color=['#2ecc71','#f39c12','#3498db'])
    ax.set_ylim(0,1)
    ax.set_ylabel("Probabilitas")
    ax.set_title("Probabilitas per Kelas")
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    chart_reader = ImageReader(buf)
    chart_w = 450
    chart_h = 300
    c.drawImage(chart_reader, (w - chart_w)/2, y_text - chart_h - 10, chart_w, chart_h)

    c.save()
    buf.close()
    return pdf_path

# ====================================================
# HALAMAN 1: Fakta & Sejarah Kaktus
# ====================================================
if page == "Fakta & Sejarah Kaktus":
    st.markdown("""
    <div style="background-color:#fdf5e6; padding:25px; border-radius:15px; margin-bottom:20px; box-shadow:0px 4px 15px rgba(0,0,0,0.05);">
        <h2 style="color:#2c6e49; text-align:center;">📖 Fakta & Sejarah Kaktus</h2>
        <p style="text-align:justify; color:#3c763d; font-size:15px;">
            Kaktus adalah tanaman dari keluarga Cactaceae, dikenal dengan kemampuan bertahan hidup di daerah gurun yang kering.
            Batangnya berdaging untuk menyimpan air dan duri menggantikan daun untuk mengurangi penguapan.
            Tanaman ini pertama kali ditemukan di Amerika dan menjadi simbol ketahanan serta keunikan alam gurun.
        </p>
        <h4 style="color:#2c6e49;">Ciri-ciri Kaktus:</h4>
        <ul style="color:#3c763d;">
            <li>Batang berdaging dan berair</li>
            <li>Duri sebagai pengganti daun</li>
            <li>Bunga indah dan beragam warna</li>
        </ul>
        <h4 style="color:#2c6e49;">Fun Facts:</h4>
        <ul style="color:#3c763d;">
            <li>Bunga kaktus bisa mekar hanya semalam</li>
            <li>Beberapa kaktus bisa hidup lebih dari 100 tahun</li>
            <li>Jenis kaktus populer: Astrophytum, Ferocactus, Gymnocalycium</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ====================================================
# HALAMAN 2: Prediksi Kaktus
# ====================================================
if page == "Prediksi Kaktus":
    #box upload image
    uploaded = st.file_uploader("Upload gambar (jpg/png)", type=["jpg","png","jpeg"])
    st.markdown("<div style='background-color:#fcf8e3; padding:25px; border-radius:14px; \
             box-shadow:0px 4px 20px rgba(0,0,0,0.06); text-align:center;'>", unsafe_allow_html=True)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    #box hasil prediksi
    st.markdown("<div style='background-color:#d9edf7; padding:15px; border-radius:10px; margin-top:15px;'>", unsafe_allow_html=True)
    st.success(f"**Prediksi: {pred_label}** ({prob:.4f})")
    st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔍 Prediksi"):
            arr = preprocess(image)
            probs = predict(interpreter, input_details, output_details, arr)

            idx = int(np.argmax(probs))
            pred_label = labels[idx]
            prob = float(probs[idx])

            st.markdown("<div style='background-color:#d9edf7; padding:15px; border-radius:10px; margin-top:15px;'>", unsafe_allow_html=True)
            st.success(f"**Prediksi: {pred_label}** ({prob:.4f})")
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("📊 Grafik Probabilitas")
            fig, ax = plt.subplots()
            ax.bar(labels, probs, color=['#2ecc71','#f39c12','#3498db'])
            ax.set_ylabel("Probabilitas")
            ax.set_ylim(0, 1)
            ax.set_title("Probabilitas per Kelas")
            st.pyplot(fig)

            st.subheader("📋 Tabel Probabilitas")
            prob_table = {
                "Kelas": labels,
                "Probabilitas": [float(p) for p in probs]
            }
            st.table(prob_table)

            pdf_path = generate_pdf(image, pred_label, probs, labels)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📄 Download Hasil Prediksi (PDF)",
                    data=f,
                    file_name="hasil_prediksi_kaktus.pdf",
                    mime="application/pdf"
                )
    else:
        st.info("Silakan upload gambar untuk memulai prediksi.")
