# ======================================================
# CATS & BIGCATS — FINAL (no text labels, manual boxes)
# ======================================================
import os, sys
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="CATS DAN BIGCATS", page_icon="🐾", layout="wide")

# -----------------------------
# DIAGNOSTIC & MODEL LOADING
# -----------------------------
def human_size(p):
    try:
        return f"{os.path.getsize(p)/1_000_000:.2f} MB"
    except Exception:
        return "N/A"

STATUS = {
    "python": sys.version.split()[0],
    "opencv": "OFF",
    "ultralytics": "OFF",
    "tensorflow": "OFF",
    "yolo_file": "missing",
    "clf_file": "missing",
    "yolo_loaded": False,
    "clf_loaded": False,
    "yolo_error": "",
    "tf_error": "",
}

yolo_model = None
classifier = None
cv2 = None
tf = None

# import cv2 if available (we need it for drawing)
try:
    import cv2
    STATUS["opencv"] = cv2.__version__
except Exception as e:
    STATUS["opencv"] = f"ERR: {e}"

# ultralytics
try:
    import ultralytics
    from ultralytics import YOLO
    STATUS["ultralytics"] = ultralytics.__version__
except Exception as e:
    YOLO = None
    STATUS["ultralytics"] = f"ERR: {e}"

# tensorflow classifier
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as keras_image
    STATUS["tensorflow"] = tf.__version__
except Exception as e:
    tf = None
    STATUS["tensorflow"] = f"ERR: {e}"

YOLO_PATH = "model/Izzatul Aliya Nisa_Laporan 4.pt"
CLF_PATH  = "model/Izzatul Aliya Nisa_Laporan 2.h5"

if os.path.exists(YOLO_PATH):
    STATUS["yolo_file"] = human_size(YOLO_PATH)
    if not str(STATUS["ultralytics"]).startswith("ERR"):
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            yolo_model = YOLO(YOLO_PATH)
            STATUS["yolo_loaded"] = True
        except Exception as e:
            STATUS["yolo_error"] = str(e)

if os.path.exists(CLF_PATH):
    STATUS["clf_file"] = human_size(CLF_PATH)
    if tf is not None and not str(STATUS["tensorflow"]).startswith("ERR"):
        try:
            classifier = tf.keras.models.load_model(CLF_PATH)
            STATUS["clf_loaded"] = True
        except Exception as e:
            STATUS["tf_error"] = str(e)

# -----------------------------
# STYLES (sama seperti sebelumnya)
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800;900&display=swap');
:root{ --red:#B31312; --red-dark:#8E0F0E; --cream:#FFF3F1; --ink:#1b1b1b; }
*{font-family:'Poppins',system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;}
[data-testid="stAppViewContainer"]{ background:var(--cream); color:var(--ink); padding-top:10px; }
[data-testid="stHeader"]{background:transparent;}
.hero{ text-align:center; margin-top:10px; margin-bottom:6px; }
.hero .t1{ font-size:76px; font-weight:900; color:var(--red); letter-spacing:.5px; }
.hero .t2{ margin-top:10px; font-size:38px; font-weight:900; color:var(--red); letter-spacing:.8px; }
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
  padding:12px; border-radius:28px; margin-bottom:22px; box-shadow:0 14px 22px rgba(179,19,18,.35); }
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

# -----------------------------
# HEADER & BADGES
# -----------------------------
st.markdown("""
<div class="hero">
  <div class="t1">CATS DAN BIGCATS</div>
  <div class="t2">DETECTION OBJEK AND CLASSIFICATION</div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f"<span class='badge'>Python {STATUS['python']}</span>"
    f"<span class='badge'>OpenCV {STATUS['opencv']}</span>"
    f"<span class='badge'>Ultralytics {STATUS['ultralytics']}</span>"
    f"<span class='badge'>TF {STATUS['tensorflow']}</span>",
    unsafe_allow_html=True
)

if STATUS["yolo_file"] != "missing":
    st.caption(f"YOLO model: {YOLO_PATH} ({STATUS['yolo_file']}) • loaded={STATUS['yolo_loaded']}")
else:
    st.caption(f"YOLO model MISSING → {YOLO_PATH}")
if STATUS["clf_file"] != "missing":
    st.caption(f"Classifier: {CLF_PATH} ({STATUS['clf_file']}) • loaded={STATUS['clf_loaded']}")
else:
    st.caption(f"Classifier MISSING → {CLF_PATH}")

if STATUS["yolo_error"]:
    st.markdown(f"<span class='badge err'>YOLO load error: {STATUS['yolo_error']}</span>", unsafe_allow_html=True)
if STATUS["tf_error"]:
    st.markdown(f"<span class='badge err'>TF load error: {STATUS['tf_error']}</span>", unsafe_allow_html=True)

# -----------------------------
# MODE & UPLOADER
# -----------------------------
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

st.markdown('<div class="uploader-wrap">', unsafe_allow_html=True)
uploaded = st.file_uploader("Drag and drop file here  •  Max 200MB  •  JPG/JPEG/PNG",
                             type=["jpg","jpeg","png"])
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# HELPER: draw boxes with OpenCV (no text)
# -----------------------------
def draw_boxes_on_image(pil_img, boxes_xyxy, class_ids=None, confidences=None):
    """
    pil_img : PIL.Image (RGB)
    boxes_xyxy : numpy array shape (N,4) with [x1,y1,x2,y2]
    class_ids : numpy array shape (N,) ints (optional)
    confidences : numpy array shape (N,) floats (optional)
    returns PIL.Image with rectangles drawn (no text)
    """
    if cv2 is None:
        # fallback: return original if cv2 not available
        return pil_img

    img = np.array(pil_img).copy()  # RGB
    # convert to BGR for cv2 drawing
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]
    thickness = max(1, int(min(h, w) / 200))  # adaptive thickness

    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [int(v) for v in box]
        cls = int(class_ids[i]) if (class_ids is not None and len(class_ids) > i) else -1

        # warna per kelas (BGR)
        # kelas 0 = Big Cats -> merah gelap
        # kelas 1 = Cats -> oranye
        if cls == 0:
            color = (18, 19, 179)  # (B,G,R) approximate dark red-ish (converted) -> pick visible
            # better choose clear colors:
            color = (10, 30, 180)  # B,G,R (strong red)
        elif cls == 1:
            color = (10, 140, 200)  # B,G,R (orangey)
        else:
            color = (200, 200, 200)

        # draw rectangle
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness=thickness, lineType=cv2.LINE_AA)

    # convert back to RGB PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

# -----------------------------
# INFERENCE (detection without text labels)
# -----------------------------
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_container_width=True)

    if st.session_state.mode == "Deteksi Objek":
        if STATUS["yolo_loaded"]:
            try:
                # run yolo
                results = yolo_model(img)

                # try to extract boxes, classes, confidences
                try:
                    # ultralytics Result object: results[0].boxes.xyxy, .cls, .conf
                    boxes = results[0].boxes.xyxy.cpu().numpy()  # shape (N,4)
                    classes = results[0].boxes.cls.cpu().numpy() if hasattr(results[0].boxes, "cls") else \
                              (results[0].boxes.cls.numpy() if hasattr(results[0].boxes, "cls") else np.array([]))
                    confs = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, "conf") else np.array([])

                    # draw our own boxes (no text)
                    img_with_boxes = draw_boxes_on_image(img, boxes, class_ids=classes, confidences=confs)
                    st.image(img_with_boxes, use_container_width=True)

                except Exception as e_inner:
                    # if that fails (structure different), fallback to plot(labels=False),
                    # but we attempted manual draw first.
                    try:
                        st.image(results[0].plot(labels=False), use_column_width=True)
                    except Exception:
                        st.error(f"Could not draw boxes: {e_inner}")
            except Exception as e:
                st.error(f"YOLO inference error: {e}")
        else:
            st.info("Deteksi belum aktif (lihat badge/status di atas).")
    else:
        # classification mode (keputusan: 0=Big Cats, 1=Cats)
        if STATUS["clf_loaded"]:
            try:
                img_res = img.resize((224, 224))
                arr = np.array(img_res).astype("float32") / 255.0
                arr = np.expand_dims(arr, axis=0)
                pred = classifier.predict(arr)
                idx = int(np.argmax(pred))
                prob = float(np.max(pred))
                label_map = {0: "Big Cats", 1: "Cats"}
                label = label_map.get(idx, "Tidak diketahui")
                st.success(f"Hasil Prediksi: **{label}**  •  Probabilitas: **{prob:.4f}**")
            except Exception as e:
                st.error(f"TF inference error: {e}")
        else:
            st.info("Klasifikasi belum aktif (lihat badge/status di atas).")

st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# GALLERY + STAT (sama seperti sebelumnya)
# -----------------------------
left, right = st.columns([1.5, 1.0], gap="large")

SEARCH_DIRS = [Path("."), Path("sample_images"), Path("images/cats"), Path("images/bigcats")]
cats_files = ["flickr_cat_000003.jpg","flickr_cat_000004.jpg","flickr_cat_000005.jpg",
              "flickr_cat_000006.jpg","flickr_cat_000008.jpg","flickr_cat_000009.jpg"]
bigcats_files = ["flickr_wild_000274.jpg","flickr_wild_000276.jpg","flickr_wild_000277.jpg",
                 "flickr_wild_000279.jpg","flickr_wild_000281.jpg","flickr_wild_000283.jpg"]

def find_images(names):
    out = []
    for n in names:
        for d in SEARCH_DIRS:
            p = d / n
            if p.exists():
                out.append(str(p))
                break
    return out

def show4(img_list):
    if not img_list:
        st.info("⚠️ Gambar belum ditemukan.")
        return
    rows = [img_list[i:i+4] for i in range(0, len(img_list), 4)]
    for row in rows:
        cols = st.columns(4)
        for i, p in enumerate(row):
            with cols[i]:
                st.image(p, use_container_width=True)

with left:
    st.markdown('<div class="section-title">Big Cats</div>', unsafe_allow_html=True)
    show4(find_images(bigcats_files))
    st.markdown("<div class='desc'><b>Big cats</b> adalah predator besar dalam keluarga Felidae seperti singa, harimau, macan tutul, jaguar, cheetah, puma, dan snow leopard.</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Cats</div>', unsafe_allow_html=True)
    show4(find_images(cats_files))
    st.markdown("<div class='desc'><b>Cats</b> adalah kucing domestik (Felis catus) yang hidup berdampingan dengan manusia.</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="stat-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="stat-header">Data yang digunakan</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-grid">
      <div><div class="stat-num">4252</div><div class="stat-label">Bigcats</div></div>
      <div><div class="stat-num">3461</div><div class="stat-label">Cats</div></div>
      <div><div class="stat-num">7713</div><div class="stat-label">All</div></div>
    </div>
    <div class="metrics">
      <div class="metric"><div class="m-title">Klasifikasi Gambar</div><div class="m-val">76%</div><div class="m-sub">Akurasi</div></div>
      <div class="metric"><div class="m-title">Deteksi Objek</div><div class="m-val">77.4%</div><div class="m-sub">Akurasi</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center;font-size:12px;color:#666;margin:28px 0 8px;">© 2025 Cats & Bigcats Dashboard — Streamlit</div>', unsafe_allow_html=True)
