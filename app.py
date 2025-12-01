# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import io
import os

st.set_page_config(page_title="Klasifikasi Jenis Kaktus", layout="centered")

@st.cache_resource
def load_tflite_model(model_path="model_kaktus.tflite"):
    """Load TFLite interpreter and metadata."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def load_class_labels(labels_path="class_labels.json"):
    """Load class labels JSON if present; otherwise use fallback list."""
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        return labels
    # fallback: try to infer from a small default ordering (edit as needed)
    return ["Astrophytum asteria", "Ferocactus", "Gymnocalycium"]

def preprocess_image_pil(img: Image.Image, target_size=(150,150)):
    """Convert PIL image to model input numpy array (float32, normalized)."""
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    # model expects shape (1, H, W, C)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_with_tflite(interpreter, input_details, output_details, image_array):
    """Run inference on preprocessed image_array (1,H,W,C)."""
    # Ensure dtype matches input details
    inp_dtype = input_details[0]["dtype"]
    # Some TFLite models expect uint8, some float32
    if inp_dtype == np.float32:
        input_data = image_array.astype(np.float32)
    else:
        # scale to 0-255 and cast
        input_data = (image_array * 255.0).astype(inp_dtype)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # If model outputs logits or probabilities per class
    probs = np.squeeze(output_data)
    # If shape is (1, n_classes) -> squeeze
    if probs.ndim == 0:
        probs = np.array([probs])
    # convert to probabilities if not already (softmax)
    try:
        # if negative values exist, apply softmax
        if np.any(probs < 0):
            exps = np.exp(probs - np.max(probs))
            probs = exps / np.sum(exps)
    except Exception:
        pass
    return probs

# --------------------
# App UI
# --------------------
st.title("Klasifikasi Jenis Kaktus 🌵")
st.write("Upload gambar kaktus, tunggu model memprediksi jenis kaktusnya.")

# Load model & labels (cache_resource so they are loaded only once per session)
with st.spinner("Memuat model..."):
    try:
        interpreter, input_details, output_details = load_tflite_model("model_kaktus.tflite")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

class_labels = load_class_labels("class_labels.json")
n_labels = len(class_labels)

uploaded_file = st.file_uploader("Pilih foto kaktus (jpg/png)", type=["jpg","jpeg","png"])
col1, col2 = st.columns(2)

if uploaded_file is not None:
    # show image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    col1.image(image, caption="Input image", use_column_width=True)
    # predict
    if st.button("Predict"):
        with st.spinner("Melakukan prediksi..."):
            try:
                img_arr = preprocess_image_pil(image, target_size=(150,150))
                probs = predict_with_tflite(interpreter, input_details, output_details, img_arr)
                # handle shape mismatch
                if probs.shape[0] != n_labels:
                    st.warning(f"Jumlah keluaran model ({probs.shape[0]}) berbeda dengan jumlah label ({n_labels}). Pastikan urutan class benar.")
                pred_idx = int(np.argmax(probs))
                pred_label = class_labels[pred_idx] if pred_idx < len(class_labels) else f"Label {pred_idx}"
                st.success(f"Prediksi: **{pred_label}** (index: {pred_idx})")
                # show probability table
                prob_items = []
                for i, p in enumerate(probs):
                    label = class_labels[i] if i < len(class_labels) else f"Label {i}"
                    prob_items.append({"class": label, "probability": float(p)})
                st.table(prob_items)
            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {e}")
else:
    st.info("Silakan upload gambar untuk memulai prediksi.")
