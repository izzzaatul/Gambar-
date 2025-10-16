import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Izzatul Aliya Nisa_Laporan 4.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/Izzatul Aliya Nisa_Laporan 2.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Info Dataset
# ==========================
jumlah_data = 824        # contoh jumlah data training
rata_rata_data = 4.7     # contoh nilai rata-rata

# ==========================
# UI Header
# ==========================
st.set_page_config(page_title="Dashboard Model", layout="wide")
st.title("📊 Dashboard Model Machine Learning")

# ==========================
# Top Section (like 824 box)
# ==========================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📈 Data Training Summary")
    col1a, col1b = st.columns(2)
    with col1a:
        st.metric(label="Jumlah Data", value=f"{jumlah_data}")
    with col1b:
        st.metric(label="Rata-rata Nilai", value=f"{rata_rata_data}")

with col2:
    st.markdown("### 🔥 Popularity Rate")
    st.metric(label="Popularity Rate", value="87%", delta="+4%")

st.divider()

# ==========================
# Mode Pilihan
# ==========================
menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        st.write("### Hasil Prediksi:", class_index)
        st.write("Probabilitas:", np.max(prediction))
