# ======================================================
# Cats & Bigcats — FINAL UI (matches red mockup)
# ======================================================
import os
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="CATS DAN BIGCATS", page_icon="🐾", layout="wide")

# ---------- OPTIONAL BACKEND (silent) ----------
# YOLO & TF dimuat diam-diam; tidak menampilkan warning bila tidak tersedia
YOLO_AVAILABLE = False
CLASSIFIER_AVAILABLE = False
yolo_model = None
classifier = None

try:
    from ultralytics import YOLO
    yolo_path = "model/Izzatul Aliya Nisa_Laporan 4.pt"
    if os.path.exists(yolo_path):
        yolo_model = YOLO(yolo_path)
        YOLO_AVAILABLE = True
except Exception:
    pass

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as keras_image
    clf_path = "model/Izzatul Aliya Nisa_Laporan 2.h5"
    if os.path.exists(clf_path):
        classifier = tf.keras.models.load_model(clf_path)
        CLASSIFIER_AVAILABLE = True
except Exception:
    pass

# ---------- STYLES (copy mockup) ----------
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
  padding-top: 10px;
}

/* remove default top bar background */
[data-testid="stHeader"]{background:transparent;}

h1,h2,h3{margin:0;padding:0}

.hero{
  text-align:center;
  margin-top:10px;
  margin-bottom:6px;
}
.hero .t1{
  font-size:76px;
  font-weight:900;
  color:var(--red);
  letter-spacing:.5px;
}
.hero .t2{
  margin-top:10px;
  font-size:38px;
  font-weight:900;
  color:var(--red);
  letter-spacing:.8px;
}

/* mode buttons (pill) */
.modebar{
  display:flex;justify-content:center;gap:22px;margin:14px 0 20px 0;
}
.modebar .btn{
  appearance:none;border:none;outline:none;cursor:pointer;
  background:var(--red); color:#fff;
  font-weight:800; letter-spacing:.3px;
  padding:14px 26px; border-radius:18px; min-width:220px;
  box-shadow:0 6px 16px rgba(179,19,18,.28);
}
.modebar .btn.alt{
  background:#fff; color:var(--red); border:2px solid var(--red);
  box-shadow:none;
}
.modebar .btn:hover{filter:brightness(.94)}
.modebar .btn.alt:hover{background:#FFF6F6}

/* uploader look */
.uploader-wrap{ max-width:820px; margin: 0 auto 20px auto; }
.uploader-wrap [data-testid="stFileUploaderDropzone"]{
  border:2px dashed #d9e6ff !important; background:#eef5ff !important;
  border-radius:16px !important;
}

/* horizontal rule */
.sep{ border:none; border-top:2px solid #E1E1E1; margin:22px 0 10px; }

/* section titles */
.section-title{
  color:var(--red); font-weight:900; font-size:28px; margin:12px 0 10px;
}

/* grid of images (exact 4 per row) */
.grid{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }
.grid img{ width:100%; height:160px; object-fit:cover; border-radius:16px;
           box-shadow:0 6px 16px rgba(0,0,0,.16); }

/* description text */
.desc{ font-size:14px; color:#3c3c3c; margin-top:10px; }

/* right red card */
.stat-card{
  background:var(--red); color:#fff; border-radius:24px; padding:22px 20px 16px;
  box-shadow:0 14px 28px rgba(179,19,18,.30);
}
.stat-header{
  background:var(--red-dark); color:#fff; text-align:center;
  font-weight:800; font-size:18px; padding:10px; border-radius:16px;
  margin-bottom:14px;
}
.stat-grid{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; text-align:center; }
.stat-num{ font-size:30px; font-weight:900; line-height:1; margin-top:4px; }
.stat-label{ font-size:13px; opacity:.95; }

/* metrics below */
.metrics{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:12px; }
.metric{ background:rgba(255,255,255,.10); border-radius:14px; padding:12px; text-align:center; }
.metric .m-title{font-weight:700}
.metric .m-val{ font-size:28px; font-weight:900; line-height:1; margin-top:4px; }
.metric .m-sub{ font-size:12px; opacity:.95 }

/* footer */
.foot{ text-align:center; font-size:12px; color:#666; margin:28px 0 8px; }
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class="hero">
  <div class="t1">CATS DAN BIGCATS</div>
  <div class="t2">DETECTION OBJEK AND CLASSIFICATION</div>
</div>
""", unsafe_allow_html=True)

# ---------- MODE BUTTONS ----------
if "mode" not in st.session_state:
    st.session_state.mode = "Deteksi Objek"

c1, c2, c3 = st.columns([1,2,1])
with c2:
    st.markdown('<div class="modebar">', unsafe_allow_html=True)
    colL, colR = st.columns(2)
    with colL:
        if st.button("Deteksi Objek", key="btn_det", use_container_width=True):
            st.session_state.mode = "Deteksi Objek"
    with colR:
        if st.button("Klasifikasi Gambar", key="btn_cls", use_container_width=True):
            st.session_state.mode = "Klasifikasi Gambar"
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- UPLOADER ----------
st.markdown('<div class="uploader-wrap">', unsafe_allow_html=True)
uploaded = st.file_uploader("Drag and drop file here  •  Max 200MB  •  JPG/JPEG/PNG",
                             type=["jpg","jpeg","png"])
st.markdown('</div>', unsafe_allow_html=True)

# ---------- PROCESS (silent; hanya tampil saat model & file ada) ----------
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption=None, use_container_width=True)
    if st.session_state.mode == "Deteksi Objek":
        if YOLO_AVAILABLE:
            res = yolo_model(img)
            st.image(res[0].plot(), caption=None, use_container_width=True)
        else:
            pass  # tetap diam; UI mengikuti mockup
    else:
        if CLASSIFIER_AVAILABLE:
            img_res = img.resize((224,224))
            from tensorflow.keras.preprocessing import image as keras_image  # noqa
            arr = np.expand_dims(keras_image.img_to_array(img_res)/255.0, axis=0)
            pred = classifier.predict(arr)
            idx = int(np.argmax(pred)); prob = float(np.max(pred))
            st.success(f"Hasil Prediksi: **{idx}**  •  Prob: **{prob:.3f}**")
        else:
            pass

# ---------- SEPARATOR ----------
st.markdown('<hr class="sep">', unsafe_allow_html=True)

# ---------- CONTENT LAYOUT ----------
left, right = st.columns([1.5, 1.0], gap="large")

# file list sesuai gambar kedua
CAT_DIR = Path("images/cats")
BIGCAT_DIR = Path("images/bigcats")
cats_files = [
    "flickr_cat_000003.jpg","flickr_cat_000004.jpg","flickr_cat_000005.jpg",
    "flickr_cat_000006.jpg","flickr_cat_000008.jpg","flickr_cat_000009.jpg"
]
bigcats_files = [
    "flickr_wild_000274.jpg","flickr_wild_000276.jpg","flickr_wild_000277.jpg",
    "flickr_wild_000279.jpg","flickr_wild_000281.jpg","flickr_wild_000283.jpg"
]

def imgs(folder: Path, files: list[str]) -> list[str]:
    return [str(folder/f) for f in files if (folder/f).exists()]

def show4(img_list: list[str]):
    # tampil 4 per baris, persis seperti mockup (tidak ada caption)
    if not img_list:
        st.info("⚠️ Gambar belum ditemukan. Pastikan file ada di folder yang benar.")
        return
    # potong jadi kelipatan 4
    rows = [img_list[i:i+4] for i in range(0, len(img_list), 4)]
    for row in rows:
        cols = st.columns(4)
        for i, p in enumerate(row):
            with cols[i]:
                st.image(p, use_container_width=True)

with left:
    # BIG CATS
    st.markdown('<div class="section-title">Big Cats</div>', unsafe_allow_html=True)
    show4(imgs(BIGCAT_DIR, bigcats_files))
    st.markdown(
        '<div class="desc"><b>Big cats</b> digunakan untuk menyebut kelompok kucing besar dalam keluarga Felidae yang merupakan predator puncak di alam liar. '
        'Mereka bertubuh besar, berotot kuat, dan efisien berburu. Termasuk singa, harimau, macan tutul, jaguar, cheetah, puma, dan snow leopard.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # CATS
    st.markdown('<div class="section-title">Cats</div>', unsafe_allow_html=True)
    show4(imgs(CAT_DIR, cats_files))
    st.markdown(
        '<div class="desc"><b>Cats</b> merujuk pada semua anggota Felidae, namun sehari-hari lebih sering untuk kucing domestik (Felis catus) '
        'yang berukuran kecil, jinak, dan hidup berdampingan dengan manusia.</div>',
        unsafe_allow_html=True
    )

with right:
    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
    st.markdown('<div class="stat-header">Data yang digunakan</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
      <div><div class="stat-num">4252</div><div class="stat-label">Bigcats</div></div>
      <div><div class="stat-num">3461</div><div class="stat-label">Cats</div></div>
      <div><div class="stat-num">7713</div><div class="stat-label">All</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metrics">
      <div class="metric"><div class="m-title">Klasifikasi Gambar</div><div class="m-val">76%</div><div class="m-sub">Akurasi</div></div>
      <div class="metric"><div class="m-title">Deteksi Objek</div><div class="m-val">77.4%</div><div class="m-sub">Akurasi</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown('<div class="foot">© 2025 Cats & Bigcats Dashboard — Streamlit</div>', unsafe_allow_html=True)
