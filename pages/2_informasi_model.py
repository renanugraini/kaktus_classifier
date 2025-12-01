import streamlit as st

st.title("📘 Informasi Model")

st.write("""
Model CNN ini dilatih menggunakan dataset 3 jenis kaktus:
- *Astrophytum asteria*
- *Ferocactus*
- *Gymnocalycium*

Augmentasi yang digunakan:
- Rotasi
- Flip horizontal & vertical
- Zoom
- Shift posisi

*Resolusi input:* 150×150  
*Epoch:* 50  
""")
