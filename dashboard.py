# ======================================================
# Cats & Bigcats — FINAL (with diagnostics for YOLO/TF)
# ======================================================
import os, sys
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="CATS DAN BIGCATS", page_icon="🐾", layout="wide")

# =========================
# DEPENDENCY & MODEL CHECKS
# =========================
def human_size(p):
    try:
        return f"{os.path.getsize(p)/1_000_000:.2f} MB"
    except Exception:
        return "N/A"

STATUS = {
    "python": sys.version.split()[0],
    "opencv": "N/A",
    "torch": "N/A",
    "ultralytics": "N/A",
    "tensorflow": "N/A",
    "yolo_model": "missing",
    "clf_model": "missing",
    "yolo_error": "",
    "tf_error": "",
}

# Try import cv2 / torch / ultralytics / tf
try:
    import cv2
    STATUS["opencv"] = cv2.__version__
except Exception as e:
    STATUS["opencv"] = f"ERR: {e}"

try:
    import torch
    STATUS["torch"] = torch.__version__
except Exception as e:
    STATUS["torch"] = f"ERR: {e}"

YOLO_AVAILABLE, CLASSIFIER_AVAILABLE = False, False
yolo_model = None
classifier = None

try:
    import ultralytics
    from ultralytics import YOLO
    STATUS["ultralytics"] = ultralytics.__version__
except Exception as e:
    STATUS["ultralytics"] = f"ERR: {e}"

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as keras_image
    STATUS["tensorflow"] = tf.__version__
except Exception as e:
    STATUS["tensorflow"] = f"ERR: {e}"
    tf = None  # type: ignore

# Paths to models
YOLO_PATH = "model/Izzatul Aliya Nisa_Laporan 4.pt"
CLF_PATH  = "model/Izzatul Aliya Nisa_Laporan 2.h5"

# Load YOLO model
if "ERR" not in STATUS["ultralytics"] and os.path.exists(YOLO_PATH):
    try:
        yolo_model = YOLO(YOLO_PATH)
        YOLO_AVAILABLE = True
        STATUS["yolo_model"] = human_size(YOLO_PATH)
    except Exception as e:
        STATUS["yolo_error"] = str(e)

# Load TF classifier
if tf is not None and os.path.exists(CLF_PATH):
    try:
        classifier = tf.keras.models.load_model(CLF_PATH)
        CLASSIFIER_AVAILABLE = True
        STATUS["clf_model"] = human_size(CLF_PATH)
    except Exception as e:
        STATUS["tf_error"] = str(e)

# =========================
# STYLES (UI merah seperti mockup)
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800;900&display=swap');
:root{ --red:#B31312; --red-dark:#8E0F0E; --cream:#FFF3F1; --ink:#1b1b1b; }
*{font-family:'Poppins',system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;}
[data-testid="stAppViewContainer"]{ background:var(--cream); color:var(--ink); padding-top:10px; }
[data-testid="stHeader"]{background:transparent;}
.hero{ text-align:center; margin-top:10px; margin-bottom:6px; }
.hero .t1{ font-size:76px; font-weight:900; color:var(--red); }
.hero .t2{ margin-top:10px; font-size:38px; font-weight:900; color:var(--red); }
#modebar .stButton>button{
  background:var(--red) !important; color:#fff !important; font-weight:800 !important;
  padding:16px 26px !important; border-radius:16px !important; border:none !important;
  min-width:260px; box-shadow:0 6px 16px rgba(179,19,18,.28); font-size:18px !important;
}
#modebar .stButton>button:hover{ background:var(--red-dark) !important; }
.uploader-wrap{ max-width:820px; margin:0 auto 20px auto; }
.uploader-wrap [data-testid="stFileUploaderDropzone"]{
  border:2px dashed #e0e0e0 !important; background:#eef2f7 !important; border-radius:16px !important;
}
.section-title{ color:var(--red); font-weight:900; font-size:28px; margin:12px 0 10px; }
.grid{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }
.grid img{ width:100%; height:160px; object-fit:cover; border-radius:16px; box-shadow:0 6px 16px rgba(0,0,0,.16); }
.desc{ font-size:14px; color:#3c3c3c; margin-top:10px; }
.stat-wrap{ background:transparent; padding:0; margin:0; }
.stat-header{ background:var(--red); color:#fff; font-weight:800; text-align:center;
  padding:12px; border-radius:28px; margin-bottom:14px; box-shadow:0 14px 22px rgba(179,19,18,.35); }
.stat-grid{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; text-align:center; margin-bottom:12px; }
.stat-num{ font-size:36px; font-weight:900; line-height:1; color:var(--red); }
.stat-label{ font-size:14px; color:var(--red); }
.metrics{ display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:10px; }
.metric{ text-align:center; padding:8px; border-radius:12px; }
.metric .m-title{ font-weight:800; color:var(--red); }
.metric .m-val{ font-size:34px; font-weight:900; margin-top:4px; color:var(--red); }
.metric .m-sub{ font-size:12px; color:var(--red); opacity:.95 }
.badge{display:inline-block;background:#fff;border:1px solid #ddd;border-radius:10px;padding:4px 8px;margin-right:6px;font-size:12px}
.badge.err{border-color:#f00;color:#b00000;background:#ffecec}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="hero">
  <div class="t1">CATS DAN BIGCATS</div>
  <div class="t2">DETECTION OBJEK AND CLASSIFICATION</div>
</div>
""", unsafe_allow_html=True)

# ===== Status ringkas agar tahu kenapa fitur “nggak jalan”
st.markdown(
    f"<span class='badge'>Python {STATUS['python']}</span>"
    f"<span class='badge'>{'OpenCV '+STATUS['opencv'] if 'ERR' not in STATUS['opencv'] else 'OpenCV: OFF'}</span>"
    f"<span class='badge'>{'Torch '+STATUS['torch'] if 'ERR' not in STATUS['torch'] else 'Torch: OFF'}</span>"
    f"<span class='badge'>{'Ultralytics '+STATUS['ultralytics'] if 'ERR' not in STATUS['ultralytics'] else 'Ultralytics: OFF'}</span>"
    f"<span class='badge'>{'TF '+STATUS['tensorflow'] if 'ERR' not in STATUS['tensorflow'] else 'TF: OFF'}</span>",
    unsafe_allow_html=True
)

if STATUS["yolo_error"]:
    st.markdown(f"<span class='badge err'>YOLO load error: {STATUS['yolo_error']}</span>", unsafe_allow_html=True)
if STATUS["tf_error"]:
    st.markdown(f"<span class='badge err'>TF load error: {STATUS['tf_error']}</span>", unsafe_allow_html=True)
if STATUS["yolo_model"] == "missing":
    st.markdown(f"<span class='badge err'>YOLO model missing: {YOLO_PATH}</span>", unsafe_allow_html=True)
if STATUS["clf_model"] == "missing":
    st.markdown(f"<span class='badge err'>Classifier model missing: {CLF_PATH}</span>", unsafe_allow_html=True)

# Mode buttons
if "mode" not in st.session_state:
    st.session_state.mode = "Deteksi Objek"
st.markdown('<div id="modebar">', unsafe_allow_html=True)
cL, cR = st.columns(2)
with cL:
    if st.button("Deteksi Objek", use_container_width=True):
        st.session_state.mode = "Deteksi Objek"
with cR:
    if st.button("Klasifikasi Gambar", use_container_width=True):
        st.session_state.mode = "Klasifikasi Gambar"
st.markdown('</div>', unsafe_allow_html=True)

# Uploader
st.markdown('<div class="uploader-wrap">', unsafe_allow_html=True)
uploaded = st.file_uploader("Drag and drop file here  •  Max 200MB  •  JPG/JPEG/PNG",
                             type=["jpg","jpeg","png"])
st.markdown('</div>', unsafe_allow_html=True)

# ==== RUN INFERENCE ====
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_container_width=True)

    if st.session_state.mode == "Deteksi Objek":
        if YOLO_AVAILABLE:
            try:
                results = yolo_model(img)
                st.image(results[0].plot(), use_container_width=True)
            except Exception as e:
                st.error(f"YOLO inference error: {e}")
        else:
            st.info("YOLO belum aktif. Pastikan Python 3.11 + torch + opencv + file .pt tersedia.")
    else:
        if CLASSIFIER_AVAILABLE:
            try:
                # >>>> Preprocessing disamakan: 224x224, float32, /255
                img_resized = img.resize((224, 224))
                arr = np.array(img_resized).astype("float32") / 255.0
                arr = np.expand_dims(arr, axis=0)
                pred = classifier.predict(arr)
                idx = int(np.argmax(pred))
                prob = float(np.max(pred))
                st.success(f"Hasil Prediksi: **{idx}**  •  Prob: **{prob:.4f}**")
            except Exception as e:
                st.error(f"TF inference error: {e}")
        else:
            st.info("Model klasifikasi belum aktif. Pastikan TensorFlow terpasang & file .h5 ada.")

