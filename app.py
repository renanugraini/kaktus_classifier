import streamlit as st
import time

st.set_page_config(
    page_title="Klasifikasi Kaktus",
    page_icon="🌵",
    layout="centered"
)

# =========================
# CSS Tema Minimalis
# =========================
st.markdown("""
<style>
body { background-color: #fafafa; }
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #333;
    text-align: center;
    margin-top: 30px;
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
    box-shadow: 0px 4px 20px rgba(0,0,0,0.05);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("<div class='main-title'>🌵 Klasifikasi Tanaman Kaktus</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Aplikasi Machine Learning berbasis Streamlit</div>", unsafe_allow_html=True)

with st.spinner("Memuat halaman..."):
    time.sleep(1)

st.write("")
st.markdown("<div class='box'>", unsafe_allow_html=True)

st.write("Selamat datang! 🎉")
st.write("""
Gunakan menu **di sidebar kiri** untuk membuka halaman:
- 📸 **Klasifikasi Gambar Kaktus**
- 📘 **Informasi Model**
""")

st.markdown("</div>", unsafe_allow_html=True)
