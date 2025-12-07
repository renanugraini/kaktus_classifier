import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

st.set_page_config(
    page_title="Cactus Classifier",
    page_icon="🌵",
    layout="centered"
)

# ============================================================
# STYLE
# ============================================================
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: 800;
            text-align: center;
            color: #2ecc71;
        }
        .subtext {
            font-size: 18px;
            text-align: justify;
        }
        .section-title {
            font-size: 24px;
            font-weight: 700;
            margin-top: 30px;
            color: #27ae60;
        }
        .box {
            background-color: #ecf9ec;
            padding: 20px;
            border-radius: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# NAVIGASI TANPA SIDEBAR
# ============================================================
page = st.selectbox(
    "Navigasi",
    ["🏠 Informasi Kaktus", "📸 Prediksi Kaktus"]
)

# ============================================================
# ---------------- HALAMAN 1: INFORMASI KAKTUS ----------------
# ============================================================
if page == "🏠 Informasi Kaktus":

    st.markdown("<div class='main-title'>🌵 Tentang Kaktus</div>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<div class='box'>", unsafe_allow_html=True)

    st.markdown("""
        <div class='section-title'>📌 Sejarah Singkat Kaktus</div>
        <p class='subtext'>
        Kaktus merupakan tanaman sukulen yang berasal dari benua Amerika dan telah beradaptasi 
        untuk hidup di lingkungan kering seperti gurun. Struktur tubuhnya mengandung cadangan air 
        yang besar dan dilindungi oleh duri, yang pada dasarnya adalah daun yang mengalami modifikasi.
        </p>

        <div class='section-title'>📌 Fakta Menarik Kaktus</div>
        <ul class='subtext'>
            <li>Kaktus memiliki lebih dari <b>2.000 spesies</b> di seluruh dunia.</li>
            <li>Duri kaktus berfungsi melindungi sekaligus mengurangi penguapan.</li>
            <li>Beberapa kaktus bisa hidup lebih dari <b>200 tahun</b>.</li>
            <li>Kaktus menghasilkan oksigen lebih banyak pada malam hari.</li>
        </ul>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.info("Klik navigasi di atas untuk menuju halaman prediksi.")


# ============================================================
# ---------------- HALAMAN 2: PREDIKSI ------------------------
# ============================================================
else:
    st.markdown("<div class='main-title'>📸 Prediksi Jenis Kaktus</div>", unsafe_allow_html=True)
    st.write("")

    uploaded = st.file_uploader("Upload gambar kaktus", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded)
        img = img.convert("RGB")

        st.image(img, caption="Gambar yang diupload", width=300)

        # ---------------------------------------------------------
        # MODEL PREDIKSI
        # ---------------------------------------------------------
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        model = tf.keras.models.load_model("model.h5")
        preds = model.predict(img_batch)[0]

        labels = ["Cactus Type A", "Cactus Type B", "Cactus Type C"]
        best_idx = np.argmax(preds)
        best_label = labels[best_idx]
        best_prob = preds[best_idx]

        st.success(f"**Prediksi: {best_label} ({best_prob*100:.2f}%)**")

        # ---------------------------------------------------------
        # GRAFIK BAR
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, preds)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probabilitas")
        ax.set_title("Probabilitas per Kelas")
        st.pyplot(fig)

        # ---------------------------------------------------------
        # DOWNLOAD PDF
        # ---------------------------------------------------------
        buf = io.BytesIO()
        pdf = canvas.Canvas(buf, pagesize=A4)

        text = f"""
        Hasil Prediksi
        ----------------------
        Prediksi : {best_label}
        Probabilitas : {best_prob*100:.2f}%
        """

        pdf.drawString(50, 780, "Hasil Prediksi Cactus Classifier")

        y = 740
        for line in text.splitlines():
            pdf.drawString(50, y, line)
            y -= 20

        # Masukkan grafik ke PDF
        chart_buf = io.BytesIO()
        fig.savefig(chart_buf, format='png')
        chart_buf.seek(0)

        chart_img = ImageReader(chart_buf)
        pdf.drawImage(chart_img, 50, 420, width=350, height=250)

        pdf.save()
        buf.seek(0)

        st.download_button(
            "Download Hasil PDF",
            data=buf,
            file_name="hasil_prediksi_kaktus.pdf",
            mime="application/pdf"
        )
