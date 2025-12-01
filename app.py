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


# ====================================================
# LOAD MODEL
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
def generate_pdf(image, pred_label, prob):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = temp.name

    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(w/2, h - 80, "Hasil Prediksi Klasifikasi Kaktus")

    # Insert image
    img_reader = ImageReader(image)
    img_w = 280
    img_h = 280
    c.drawImage(img_reader, (w - img_w) / 2, h - 380, img_w, img_h)

    # Prediction text
    c.setFont("Helvetica", 14)
    c.drawString(80, h - 420, f"Prediksi : {pred_label}")
    c.drawString(80, h - 440, f"Akurasi : {prob:.4f}")

    c.save()
    return pdf_path


# ====================================================
# UI
# ====================================================
st.markdown("<div class='main-title'>🌵 Klasifikasi Tanaman Kaktus</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Upload gambar kaktus dan lihat prediksinya</div>", unsafe_allow_html=True)
st.write("")

with st.spinner("Memuat model..."):
    interpreter, input_details, output_details = load_tflite_model()

labels = load_class_labels()

uploaded = st.file_uploader("Upload gambar (jpg/png)", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded)

    st.markdown("<div class='box'>", unsafe_allow_html=True)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Prediksi"):
        arr = preprocess(image)
        probs = predict(interpreter, input_details, output_details, arr)

        idx = int(np.argmax(probs))
        pred_label = labels[idx]
        prob = float(probs[idx])

        st.success(f"**Prediksi: {pred_label}** ({prob:.4f})")

        # PDF GENERATION
        pdf_path = generate_pdf(image, pred_label, prob)

        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Download Hasil Prediksi (PDF)",
                data=f,
                file_name="hasil_prediksi_kaktus.pdf",
                mime="application/pdf"
            )

else:
    st.info("Silakan upload gambar untuk memulai prediksi.")
