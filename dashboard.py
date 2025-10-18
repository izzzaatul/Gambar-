# ======================================================
# Image Classification & Object Detection Dashboard — Red Mockup
# ======================================================
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from ultralytics import YOLO
import os
from pathlib import Path

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="CATS dan BIGCATS — Detection & Classification",
    page_icon="🐾",
    layout="wide"
)

# ---------- THEME & STYLES (Merah seperti mockup) ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700;800&display=swap');

:root{
  --red:#B31312;
  --red-dark:#870E0D;
  --cream:#FFF7F5;
  --text:#1b1b1b;
}

* { font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }

[data-testid="stAppViewContainer"]{
  background: var(--cream);
  color: var(--text);
}

h1,h2,h3{ color: var(--red); font-weight:800; letter-spacing: 1px; }

.header-title{
  text-align:center; font-size:64px; line-height:1.05; margin-top:8px; margin-bottom:4px;
}
.sub-title{
  text-align:center; font-size:52px; line-height:1.05; margin-top:0; margin-bottom:24px;
}

.center-row{ display:flex; align-items:center; justify-content:center; gap:16px; margin: 8px 0 16px 0; }
.app-btn{
  background: var(--red); border:none; color:white; padding:14px 22px; border-radius:14px;
  font-weight:700; letter-spacing:.2px; cursor:pointer;
}
.app-btn:hover{ background: var(--red-dark); }

.upload-box > div{ border-radius:16px !important; border:2px dashed #e0e0e0 !important; }

.section-title{ font-size:28px; color:var(--red); font-weight:800; margin-top:8px; }
.section-desc{ font-size:14px; color:#403f3f; margin-top:6px; }

.grid{ display:grid; grid-template-columns: repeat(4, 1fr); gap:16px; }
.grid img{ width:100%; height:160px; object-fit:cover; border-radius:18px;
           box-shadow:0 4px 14px rgba(0,0,0,.12); }

.stat-card{
  background: var(--red);
  color:white;
  border-radius:24px;
  padding:28px;
  box-shadow: 0 12px 30px rgba(179,19,18,.25);
}
.stat-head{ font-size:26px; font-weight:800; text-align:center; margin-bottom:10px; }
.stat-grid{ display:grid; grid-template-columns: repeat(3,1fr); gap:14px; margin-top:6px; }
.stat-box{ background: rgba(255,255,255,.08); padding:12px; border-radius:14px; text-align:center; }
.stat-num{ font-size:30px; font-weight:800; line-height:1; }
.stat-label{ font-size:14px; opacity:.95; }

.metric-wrap{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:12px; }
.metric{ background: rgba(255,255,255,.08); padding:14px; border-radius:14px; text-align:center; }
.metric .big{ font-size:30px; font-weight:800; }
.metric .small{ font-size:14px; }

.footer{ text-align:center; font-size:13px; color:#5a5a5a; margin-top:24px; }
</style>
""", unsafe_allow_html=True)

# ---------- MODEL LOADING ----------
@st.cache_resource
def load_models():
    base_path = "model"
    yolo_path = os.path.join(base_path, "Izzatul Aliya Nisa_Laporan 4.pt")
    classifier_path = os.path.join(base_path, "Izzatul Aliya Nisa_Laporan 2.h5")

    if not os.path.exists(yolo_path):
        st.error(f"❌ File YOLO tidak ditemukan di: {yolo_path}")
        return None, None
    if not os.path.exists(classifier_path):
        st.error(f"❌ File Klasifikasi tidak ditemukan di: {classifier_path}")
        return None, None

    yolo_model = YOLO(yolo_path)
    classifier = tf.keras.models.load_model(classifier_path)
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ---------- HEADER (sesuai mockup) ----------
st.markdown("<div class='header-title'>CATS DAN BIGCATS</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>DETECTION OBJEK AND CLASSIFICATION</div>", unsafe_allow_html=True)

# ---------- MODE SWITCH (tombol seperti mockup) ----------
if "mode" not in st.session_state:
    st.session_state.mode = "Deteksi Objek (YOLO)"

colA, colB, colC = st.columns([1,1,1], vertical_alignment="center")
with colA:
    pass
with colB:
    c1, c2 = st.columns(2)
    if c1.button("Deteksi Objek", use_container_width=True, type="primary"):
        st.session_state.mode = "Deteksi Objek (YOLO)"
    if c2.button("Klasifikasi Gambar", use_container_width=True):
        st.session_state.mode = "Klasifikasi Gambar"
with colC:
    pass

# ---------- UPLOADER (di tengah) ----------
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop file here  •  Max 200MB  •  JPG/JPEG/PNG", type=["jpg","jpeg","png"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_container_width=True)
    if yolo_model is not None and classifier is not None:
        if st.session_state.mode == "Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek")
            results = yolo_model(img)
            result_img = results[0].plot()  # numpy array (BGR)
            st.image(result_img, caption="Hasil Deteksi (YOLO)", use_container_width=True)
        else:
            st.subheader("🧠 Hasil Klasifikasi Gambar")
            img_resized = img.resize((224, 224))
            arr = keras_image.img_to_array(img_resized)
            arr = np.expand_dims(arr, axis=0) / 255.0
            pred = classifier.predict(arr)
            idx = int(np.argmax(pred))
            prob = float(np.max(pred))
            st.success(f"Hasil Prediksi: **{idx}**")
            st.write(f"Probabilitas: **{prob:.4f}**")
    else:
        st.warning("⚠️ Model belum berhasil dimuat. Periksa kembali folder `model/`.")

st.markdown("---")

# =====================================================
# GALLERY DATA (ganti kotak hitam dengan gambar dari daftar “foto ke-2”)
# =====================================================
# Atur path folder gambarnya sesuai struktur kamu
CAT_DIR = Path("images/cats")
BIGCAT_DIR = Path("images/bigcats")

# Daftar file contoh (dari screenshot kedua). Taruh file2 ini ke folder di atas.
cats_files = [
    "flickr_cat_000003.jpg", "flickr_cat_000004.jpg", "flickr_cat_000005.jpg",
    "flickr_cat_000006.jpg", "flickr_cat_000008.jpg", "flickr_cat_000009.jpg"
]
bigcats_files = [
    "flickr_wild_000274.jpg", "flickr_wild_000276.jpg", "flickr_wild_000277.jpg",
    "flickr_wild_000279.jpg", "flickr_wild_000281.jpg", "flickr_wild_000283.jpg"
]

def load_existing_images(base: Path, names: list[str]) -> list[str]:
    paths = []
    for n in names:
        p = base / n
        if p.exists():
            paths.append(str(p))
    return paths

# Grid helper
def show_grid(images: list[str], per_row: int = 4, height: int = 160):
    if not images:
        st.info("⚠️ Gambar belum ditemukan. Pastikan file-file ada di folder `images/...` seperti pada daftar nama file.")
        return
    # tampil grid manual agar konsisten
    rows = [images[i:i+per_row] for i in range(0, len(images), per_row)]
    for row in rows:
        cols = st.columns(per_row)
        for i, src in enumerate(row):
            with cols[i]:
                st.image(src, use_container_width=True)

# ====== Layout dua kolom besar: kiri (konten), kanan (stat-card)
left, right = st.columns([1.4, 0.9], gap="large")

with left:
    # BIG CATS
    st.markdown("<div class='section-title'>Big Cats</div>", unsafe_allow_html=True)
    show_grid(load_existing_images(BIGCAT_DIR, bigcats_files), per_row=4)

    st.markdown(
        "<div class='section-desc'><b>Big cats</b> adalah kelompok kucing besar dalam keluarga Felidae yang menjadi predator puncak. "
        "Ciri utamanya bertubuh besar, berotot kuat, serta kemampuan berburu yang efisien. "
        "Contoh: singa, harimau, macan tutul, jaguar, cheetah, puma, dan snow leopard.</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # CATS (domestik)
    st.markdown("<div class='section-title'>Cats</div>", unsafe_allow_html=True)
    show_grid(load_existing_images(CAT_DIR, cats_files), per_row=4)

    st.markdown(
        "<div class='section-desc'><b>Cats</b> merujuk pada semua Felidae, namun sehari-hari dipakai untuk kucing domestik (Felis catus). "
        "Kucing domestik berukuran relatif kecil, jinak, dan hidup berdampingan dengan manusia sebagai hewan peliharaan.</div>",
        unsafe_allow_html=True
    )

with right:
    # STAT CARD (merah besar di kanan)
    st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
    st.markdown("<div class='stat-head'>Data yang digunakan</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-num">4252</div>
        <div class="stat-label">Bigcats</div>
      </div>
      <div class="stat-box">
        <div class="stat-num">3461</div>
        <div class="stat-label">Cats</div>
      </div>
      <div class="stat-box">
        <div class="stat-num">7713</div>
        <div class="stat-label">All</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-wrap">
      <div class="metric">
        <div>Klasifikasi Gambar</div>
        <div class="big">76%</div>
        <div class="small">Akurasi</div>
      </div>
      <div class="metric">
        <div>Deteksi Objek</div>
        <div class="big">77.4%</div>
        <div class="small">Akurasi</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("<div class='footer'>© 2025 Cats & Bigcats Dashboard — Streamlit</div>", unsafe_allow_html=True)
