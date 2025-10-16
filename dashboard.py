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
# Custom CSS
# ==========================
st.markdown(
    """
    <style>
        /* Background dan teks utama */
        .main {
            background-color: #f4f7fa;
            color: #2b2d42;
        }

        /* Judul */
        h1 {
            color: #1e3a8a;
            text-align: center;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #1e293b;
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }

        /* Tombol upload */
        div.stFileUploader label {
            background-color: #2563eb;
            color: white !important;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
        }
        div.stFileUploader label:hover {
            background-color: #1d4ed8;
        }

        /* Garis pemisah */
        hr {
            border: 1px solid #cbd5e1;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("📋 Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="🟩 Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)

        st.markdown(f"### 🧾 **Hasil Prediksi:** {class_index}")
        st.markdown(f"**Probabilitas:** `{np.max(prediction):.4f}`")
