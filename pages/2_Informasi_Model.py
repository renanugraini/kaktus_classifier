import streamlit as st

st.title("📘 Informasi Model")

st.write("""
Model ini dilatih menggunakan CNN dengan dataset 3 jenis kaktus:
- **Astrophytum asteria**
- **Ferocactus**
- **Gymnocalycium**

Dengan augmentasi:
- Rotasi
- Flip horizontal & vertical
- Zoom
- Shift posisi

Resolution input: **150x150**  
Epoch: **50**
""")
