import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("📸 Klasifikasi Gambar Kaktus")

uploaded = st.file_uploader("Upload gambar kaktus...", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).resize((150,150))
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # Load model hanya sekali
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model("model.h5")

    model = load_model()

    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    pred = model.predict(img_array)[0]

    labels = ["Astrophytum asteria", "Ferocactus", "Gymnocalycium"]

    hasil = labels[np.argmax(pred)]
    prob = np.max(pred)

    st.success(f"*Prediksi: {hasil}* ({prob:.4f})")
