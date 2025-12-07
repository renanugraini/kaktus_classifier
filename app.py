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
st.set_page_config(page_title="Klasifikasi Kaktus", page_icon="🌵", layout="centered")

# ====================================================
# CUSTOM STYLE
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

/* Light mode default */
body, .stApp {
    background-color: var(--soft-green);
    color: var(--text-light);
    font-family: 'Segoe UI', sans-serif;
}

/* Box style */
.box, .menu-box {
    padding: 25px;
    border-radius: 14px;
    background: white;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.07);
    transition: 0.3s ease;
}

/* Hover effect cards */
.box:hover, .menu-box:hover {
    box-shadow: 0px 6px 25px rgba(0,0,0,0.12);
}

/* Titles */
h1, h2, h3 {
    color: var(--cactus-green-dark);
    font-weight: bold;
}

/* Centered image */
.centered-img {
    display: flex;
    justify-content: center;
    margin-bottom: 10px;
}

/* Selectbox styling */
div[data-baseweb="select"] > div {
    background-color: white !important;
    border-radius: 10px !important;
    border: 1.5px solid var(--cactus-green) !important;
}

/* Dark Mode Auto */
@media (prefers-color-scheme: dark) {
    body, .stApp {
        background-color: var(--dark-bg) !important;
        color: var(--text-dark) !important;
    }

    .box, .menu-box {
        background: var(--dark-card) !important;
        color: var(--text-dark) !important;
        box-shadow: 0px 4px 20px rgba(255,255,255,0.05);
    }

    h1, h2, h3 {
        color: var(--cactus-green) !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #2c2c2c !important;
        border: 1.5px solid var(--cactus-green) !important;
        color: white !important;
    }
}

/* Button styling */
.stButton>button {
    background: var(--cactus-green) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    border: none !important;
    font-size: 16px !important;
}
.stButton>button:hover {
    background: var(--cactus-green-dark) !important;
}

</style>
""", unsafe_allow_html=True)

</style>
""", unsafe_allow_html=True)

# ====================================================
# LOAD MODEL TFLITE
# ====================================================
@st.cache_resource
def load_tflite_model(model_path="model_kaktus.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

# Load labels
def load_class_labels(path="class_labels.json"):
    if os.path.exists(path):
        return json.load(open(path, "r"))
    return ["Astrophytum asteria", "Ferocactus", "Gymnocalycium"]

# ====================================================
# PREPROCESS
# ====================================================
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((150,150))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

# ====================================================
# PREDICT
# ====================================================
def predict(interpreter, input_details, output_details, array):
    interpreter.set_tensor(input_details[0]["index"], array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return np.squeeze(output)

# ====================================================
# GENERATE PDF
# ====================================================
def generate_pdf(image, pred_label, probs, labels):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = temp.name

    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(w/2, h - 80, "Hasil Prediksi Klasifikasi Kaktus")

    # Uploaded image (small & centered)
    img_for_pdf = image.copy().convert("RGB")
    img_reader = ImageReader(img_for_pdf)
    img_w = 200
    img_h = 200
    c.drawImage(img_reader, (w - img_w)/2, h - 120 - img_h, img_w, img_h)

    # Prediction text
    c.setFont("Helvetica-Bold", 14)
    y_text = h - 120 - img_h - 25
    c.drawCentredString(w/2, y_text, f"Prediksi : {pred_label}")

    # Probability table
    c.setFont("Helvetica", 12)
    y_text -= 25
    for i, p in enumerate(probs):
        c.drawCentredString(w/2, y_text, f"{labels[i]} : {p:.4f}")
        y_text -= 18

    # Bar chart
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(14,4))
    ax.bar(labels, probs, color=['#2ecc71','#f39c12','#3498db'])
    ax.set_ylim(0,1)
    ax.set_ylabel("Probabilitas")
    ax.set_title("Probabilitas per Kelas")
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)

    chart_reader = ImageReader(buf)
    chart_w = 480
    chart_h = 300
    c.drawImage(chart_reader, (w - chart_w)/2, y_text - chart_h - 10, chart_w, chart_h)

    c.save()
    buf.close()
    return pdf_path


# ====================================================
# MENU (2 Halaman Tanpa Sidebar)
# ====================================================
menu = st.selectbox("Menu", ["🏠 Tentang Kaktus", "🌵 Prediksi Kaktus"])

labels = load_class_labels()

with st.spinner("Memuat model..."):
    interpreter, input_details, output_details = load_tflite_model("model_kaktus.tflite")

# ====================================================
# HALAMAN 1 — Tentang Kaktus
# ====================================================
if menu == "🏠 Tentang Kaktus":

    st.markdown("<div class='menu-box'>", unsafe_allow_html=True)
    st.title("🌵 Tentang Tanaman Kaktus")
    st.write("")

    st.write("""
    Kaktus adalah tanaman sukulen yang berasal dari daerah gurun dan dikenal mampu 
    menyimpan air di batangnya. Tanaman ini memiliki bentuk unik, pertahanan berupa 
    duri, dan mampu bertahan hidup di kondisi ekstrim.
    """)

    st.subheader("✨ Fakta Menarik tentang Kaktus")
    st.write("""
    - Kaktus dapat menyimpan air hingga **200 liter** di batangnya.
    - Duri pada kaktus berfungsi melindungi dari hewan dan mengurangi penguapan.
    - Kaktus termasuk tanaman yang berfotosintesis menggunakan metode **CAM**, 
      yang membuatnya efisien bertahan di cuaca panas.
    - Beberapa kaktus dapat hidup hingga **200 tahun**.
    """)

    st.markdown("</div>", unsafe_allow_html=True)



# ====================================================
# HALAMAN 2 — Prediksi Kaktus
# ====================================================
elif menu == "🌵 Prediksi Kaktus":
st.markdown("<h1 style='text-align:center;'>🌵 Prediksi Jenis Kaktus</h1>", unsafe_allow_html=True)

    st.title("🌵 Prediksi Jenis Kaktus")
    uploaded = st.file_uploader("Upload gambar (jpg/png)", type=["jpg","png","jpeg"])

    if uploaded:
        image = Image.open(uploaded)

        st.markdown("<div class='centered-img'>", unsafe_allow_html=True)
        st.image(image, width=300)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔍 Prediksi"):
            arr = preprocess(image)
            probs = predict(interpreter, input_details, output_details, arr)
            idx = int(np.argmax(probs))
            pred_label = labels[idx]

            st.success(f"**Prediksi: {pred_label}**")

            # Grafik
            fig, ax = plt.subplots()
            ax.bar(labels, probs)
            ax.set_ylim(0,1)
            ax.set_ylabel("Probabilitas")
            st.pyplot(fig)

            # Download PDF
            pdf_path = generate_pdf(image, pred_label, probs, labels)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📄 Download PDF",
                    data=f,
                    file_name="hasil_prediksi_kaktus.pdf",
                    mime="application/pdf"
                )

    else:
        st.info("Silakan upload gambar untuk memulai prediksi.")
