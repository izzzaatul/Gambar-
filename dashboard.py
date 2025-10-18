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
  padding-top:10px;
}
[data-testid="stHeader"]{background:transparent;}
h1,h2,h3{margin:0;padding:0}

/* HERO */
.hero{ text-align:center; margin-top:10px; margin-bottom:6px; }
.hero .t1{ font-size:76px;
