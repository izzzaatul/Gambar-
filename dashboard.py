# ======================================================
# Cats & Bigcats — FINAL RED MOCKUP (Fully Fixed)
# ======================================================
import os
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="CATS DAN BIGCATS", page_icon="🐾", layout="wide")

# ---------- BACKEND (silent, optional) ----------
YOLO_AVAILABLE = False
CLASSIFIER_AVAILABLE = False
yolo_model = None
classifier = None

try:
    from ultralytics import YOLO
    if os.path.exists("model/Izzatul Aliya Nisa_Laporan 4.pt"):
        yolo_model = YOLO("model/Izzatul Aliya Nisa_Laporan 4.pt")
        YOLO_AVAILABLE = True
except Exception:
    pass

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as keras_image
    if os.path.exists("model/Izzatul Aliya Nisa_Laporan 2.h5"):
        classifier = tf.keras.models.load_model("model/Izzatul Aliya Nisa_Laporan 2.h5")
        CLASSIFIER_AVAILABLE = True
except Exception:
    pass

# ---------- STYLE ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800;900&display=swap');

:root{
  --red:#B31312;
  --red-dark:#8E0F0E;
  --cream:#FFF3F1;
  --ink:#1b1b1b;
}

*{font-family:'Poppins',system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;}

[data-testid="stAppViewContainer"]{
  background:var(--cream);
  color:var(--ink);
  padding-top:10px;
}
[data-testid="stHeader"]{background:transparent;}

/* HERO */
.hero{text-align:center;margin-top:10px;margin-bottom:6px;}
.hero .t1{font-size:76px;font-weight:900;color:var(--red);letter-spacing:.5px;}
.hero .t2{margin-top:10px;font-size:38px;font-weight:900;color:var(--red);letter-spacing:.8px;}

/* MODE BUTTONS: Merah, teks putih */
#modebar .stButton>button{
  background:var(--red) !important;
  color:#fff !important;
  font-weight:800 !important;
  letter-spacing:.3px;
  padding:14px 26px !important;
  border-radius:18px !important;
  border:none !important;
  min-width:220px;
  box-shadow:0 6px 16px rgba(179,19,18,.28);
}
#modebar .stButton>button:hover{
  background:var(--red-dark) !important;
}

/* UPLOADER */
.uploader-wrap{max-width:820px;margin:0 auto 20px auto;}
.uploader-wrap [data-testid="stFileUploaderDropzone"]{
  border:2px dashed #e0e0e0 !important;
  background:#eef2f7 !important;
  border-radius:16px !important;
}

/* HR */
.sep{border:none;border-top:2px solid #E1E1E1;margin:22px 0 10px;}

/* SECTION */
.section-title{color:var(--red);font-weight:900;font-size:28px;margin:12px 0 10px;}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;}
.grid img{width:100%;height:160px;object-fit:cover;border-radius:16px;
          box-shadow:0 6px 16px rgba(0,0,0,.16);}
.desc{font-size:14px;color:#3c3c3c;margin-top:10px;}

/* STAT CARD */
.stat-card{
  background:#fff;
  color:var(--red);
  border-radius:24px;
  padding:20px;
  box-shadow:0 12px 24px rgba(0,0,0,.1);
}
.stat-header{
  background:var(--red);
  color:#fff;
  font-weight:800;
  text-align:center;
  padding:10px;
  border-radius:25px;
  margin-bottom:16px;
  box-shadow:0 10px 16px rgba(179,19,18,.35);
}
.stat-grid{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  text-align:center;
  gap:10px;
  margin-bottom:12px;
}
.stat-num{font-size:30px;font-weight:900;line-height:1;}
.stat-label{font-size:13px;}
.metrics{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:10px;
  margin-top:10px;
}
.metric{
  text-align:center;
  padding:10px;
  border-radius:12px;
}
.metric .m-title{font-weight:700;}
.metric .m-val{font-size:28px;font-weight:900;margin-top:4px;}
.metric .m-sub{font-size:12px;opacity:.9;}

/* FOOTER */
.foot{text-align:center;font-size:12px;color:#666;margin:28px 0 8px;}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="hero">
  <div class="t1">CATS DAN BIGCATS</div>
  <div class="t2">DETECTION OBJEK AND CLASSIFICATION</div>
</div>
""", unsafe_allow_html=True)

# ---------- MODE BUTTONS ----------
if "mode" not in st.session_state:
    st.session_state.mode = "Deteksi Objek"

st.markdown('<div id="modebar">', unsafe_allow_html=True)
colL, colR = st.columns(2)
with colL:
    if st.button("Deteksi Objek", use_container_width=True):
        st.session_state.mode = "Deteksi Objek"
with colR:
    if st.button("Klasifikasi Gambar", use_container_width=True):
        st.session_state.mode = "Klasifikasi Gambar"
st.markdown('</div>', unsafe_allow_html=True)

# ---------- UPLOADER ----------
st.markdown('<div class="uploader-wrap">', unsafe_allow_html=True)
uploaded = st.file_uploader("Drag and drop file here  •  Max 200MB  •  JPG/JPEG/PNG",
                             type=["jpg","jpeg","png"])
st.markdown('</div>', unsafe_allow_html=True)

# ---------- PROCESS ----------
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_container_width=True)
    if st.session_state.mode == "Deteksi Objek" and YOLO_AVAILABLE:
        res = yolo_model(img)
        st.image(res[0].plot(), use_container_width=True)
    elif st.session_state.mode == "Klasifikasi Gambar" and CLASSIFIER_AVAILABLE:
        img_res = img.resize((224,224))
        arr = np.expand_dims(np.array(img_res)/255.0, axis=0)
        pred = classifier.predict(arr)
        st.success(f"Hasil Prediksi: {np.argmax(pred)}")

# ---------- SEPARATOR ----------
st.markdown('<hr class="sep">', unsafe_allow_html=True)

# ---------- CONTENT ----------
left, right = st.columns([1.5, 1.0], gap="large")

CAT_DIR = Path("images/cats")
BIGCAT_DIR = Path("images/bigcats")
cats_files = ["flickr_cat_000003.jpg","flickr_cat_000004.jpg","flickr_cat_000005.jpg",
              "flickr_cat_000006.jpg","flickr_cat_000008.jpg","flickr_cat_000009.jpg"]
bigcats_files = ["flickr_wild_000274.jpg","flickr_wild_000276.jpg","flickr_wild_000277.jpg",
                 "flickr_wild_000279.jpg","flickr_wild_000281.jpg","flickr_wild_000283.jpg"]

def imgs(folder: Path, files: list[str]) -> list[str]:
    return [str(folder/f) for f in files if (folder/f).exists()]

def show4(img_list: list[str]):
    if not img_list:
        st.info("⚠️ Gambar belum ditemukan. Pastikan file ada di folder yang benar.")
        return
    rows = [img_list[i:i+4] for i in range(0, len(img_list), 4)]
    for row in rows:
        cols = st.columns(4)
        for i, p in enumerate(row):
            with cols[i]:
                st.image(p, use_container_width=True)

with left:
    st.markdown('<div class="section-title">Big Cats</div>', unsafe_allow_html=True)
    show4(imgs(BIGCAT_DIR, bigcats_files))
    st.markdown("<div class='desc'><b>Big cats</b> adalah kelompok kucing besar ...</div>", unsafe_allow_html=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Cats</div>', unsafe_allow_html=True)
    show4(imgs(CAT_DIR, cats_files))
    st.markdown("<div class='desc'><b>Cats</b> adalah kucing domestik ...</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
    st.markdown('<div class="stat-header">Data yang digunakan</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
      <div><div class="stat-num">4252</div><div class="stat-label">Bigcats</div></div>
      <div><div class="stat-num">3461</div><div class="stat-label">Cats</div></div>
      <div><div class="stat-num">7713</div><div class="stat-label">All</div></div>
    </div>
    <div class="metrics">
      <div class="metric">
        <div class="m-title">Klasifikasi Gambar</div>
        <div class="m-val">76%</div>
        <div class="m-sub">Akurasi</div>
      </div>
      <div class="metric">
        <div class="m-title">Deteksi Objek</div>
        <div class="m-val">77.4%</div>
        <div class="m-sub">Akurasi</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown('<div class="foot">© 2025 Cats & Bigcats Dashboard — Streamlit</div>', unsafe_allow_html=True)
