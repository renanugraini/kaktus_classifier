import streamlit as st

st.title("📘 Tentang Model")
st.write("""
Model ini dibuat menggunakan TensorFlow dan merupakan model klasifikasi gambar dengan 3 kelas kaktus:

- **Astrophytum asteria**
- **Ferocactus**
- **Gymnocalycium**

Model dilatih dengan teknik augmentasi data:

- Rotasi 40°
- Shift horizontal & vertikal
- Zoom 30%
- Flip horizontal & vertical

Dan dikonversi menjadi **TFLite** agar ringan saat dipakai di Streamlit Cloud.
""")

st.info("Jika ingin memperbarui model, cukup upload file `.tflite` terbaru ke repository.")
