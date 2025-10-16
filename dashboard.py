# ======================================================
# Image Classification & Object Detection Dashboard (Full)
# ======================================================
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from ultralytics import YOLO

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Image Classification & Object Detection",
    page_icon="🧠",
    layout="wide"
)

# ---------- CUSTOM BACKGROUND ----------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom, #d4f8d4 60%, #d4f8d4 80%, #e6d4f8 100%);
    color: #1b1b1b;
    font-family: "Poppins", sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

h1, h2, h3 {
    font-weight: 600;
    color: #1b4332;
}

p {
    font-size: 16px;
    line-height: 1.6;
}

img {
    border-radius: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Izzatul Aliya Nisa_Laporan 4.pt")  # Deteksi objek
    classifier = tf.keras.models.load_model("model/Izzatul Aliya Nisa_Laporan 2.h5")  # Klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ---------- HEADER ----------
st.markdown("<h1 style='text-align: center;'>Image Classification & Object Detection</h1>", unsafe_allow_html=True)
st.write("")

# =====================================================
# BIG CATS SECTION
# =====================================================
st.markdown("## 🦁 Big Cats")

col1, col2 = st.columns([1, 1.8])
with col1:
    st.image([
        "https://cdn-icons-png.flaticon.com/512/616/616408.png",
        "https://cdn-icons-png.flaticon.com/512/616/616425.png",
        "https://cdn-icons-png.flaticon.com/512/616/616430.png"
    ], caption=["Lion", "Leopard", "Tiger"], width=180)
with col2:
    st.markdown("""
    **Big cats** digunakan untuk menyebut kelompok kucing besar dalam keluarga *Felidae* yang merupakan predator puncak di alam liar.  
    Mereka memiliki tubuh besar, kekuatan luar biasa, serta kemampuan berburu yang efisien.  
    Termasuk di antaranya **singa, harimau, macan tutul, jaguar, cheetah, puma**, dan **snow leopard**.  
    """)

# =====================================================
# DOMESTIC CATS SECTION
# =====================================================
st.markdown("---")
st.markdown("## 🐱 Cats")

col3, col4 = st.columns([1, 1.8])
with col3:
    st.image([
        "https://cdn-icons-png.flaticon.com/512/2206/2206368.png",
        "https://cdn-icons-png.flaticon.com/512/616/616408.png"
    ], caption=["Domestic Cat 1", "Domestic Cat 2"], width=180)
with col4:
    st.markdown("""
    **Cats** merujuk pada semua anggota keluarga *Felidae*, namun dalam penggunaan sehari-hari lebih sering digunakan untuk menyebut **kucing domestik (*Felis catus*)**.  
    Kucing domestik berukuran kecil, bersifat jinak, dan hidup berdampingan dengan manusia sebagai hewan peliharaan.
    """)

# =====================================================
# MODEL SECTION — UPLOAD & PREDIKSI
# =====================================================
st.markdown("---")
st.markdown("## 📸 Coba Deteksi atau Klasifikasi Sendiri!")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("Unggah gambar di sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        st.write("### 🔍 Hasil Deteksi Objek")
        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi (YOLO)", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        st.write("### 🧠 Hasil Klasifikasi Gambar")
        img_resized = img.resize((224, 224))
        img_array = keras_image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        st.success(f"Hasil Prediksi: **{class_index}**")
        st.write("Probabilitas:", np.max(prediction))

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("<p style='text-align:center; font-size:14px; color:#444;'>© 2025 Image Classification Dashboard — created with Streamlit</p>", unsafe_allow_html=True)
