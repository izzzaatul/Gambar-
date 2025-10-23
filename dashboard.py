# ======================================================
# CATS & BIGCATS — FINAL (STRICT: NO TEXT on YOLO boxes)
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

# try import cv2
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
# tensorflow
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
# BASIC STYLES (kept minimal)
# -----------------------------
st.markdown("""
<style>
:root{ --red:#B31312; --cream:#FFF3F1; --ink:#1b1b1b; }
body { background:var(--cream); color:var(--ink); }
</style>
""", unsafe_allow_html=True)

st.title("CATS DAN BIGCATS — Detection & Classification")
st.text(f"Python {STATUS['python']}  |  OpenCV {STATUS['opencv']}  |  Ultralytics {STATUS['ultralytics']}  |  TF {STATUS['tensorflow']}")
if STATUS["yolo_file"] != "missing":
    st.caption(f"YOLO model: {YOLO_PATH} ({STATUS['yolo_file']}) • loaded={STATUS['yolo_loaded']}")
else:
    st.caption(f"YOLO model MISSING → {YOLO_PATH}")
if STATUS["clf_file"] != "missing":
    st.caption(f"Classifier: {CLF_PATH} ({STATUS['clf_file']}) • loaded={STATUS['clf_loaded']}")
else:
    st.caption(f"Classifier MISSING → {CLF_PATH}")

# -----------------------------
# UPLOADER & MODE
# -----------------------------
if "mode" not in st.session_state:
    st.session_state.mode = "Deteksi Objek"

col1, col2 = st.columns(2)
with col1:
    if st.button("Deteksi Objek", use_container_width=True):
        st.session_state.mode = "Deteksi Objek"
with col2:
    if st.button("Klasifikasi Gambar", use_container_width=True):
        st.session_state.mode = "Klasifikasi Gambar"

uploaded = st.file_uploader("Upload JPG/PNG (Max 200MB)", type=["jpg","jpeg","png"])

# -----------------------------
# HELPER: draw boxes (NO TEXT)
# -----------------------------
def draw_boxes_only(pil_img, boxes_xyxy, class_ids=None, thickness_ratio=200):
    """
    Draw only rectangles (no text) and return PIL.Image.
    boxes_xyxy: (N,4) array with x1,y1,x2,y2 in pixel coords
    class_ids: optional list/array of ints (same length as boxes)
    """
    if cv2 is None:
        return pil_img

    img = np.array(pil_img).copy()  # RGB uint8
    # convert to BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    thickness = max(1, int(min(h, w) / thickness_ratio))

    # ensure arrays
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return pil_img

    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [int(v) for v in box]
        cls = int(class_ids[i]) if (class_ids is not None and i < len(class_ids)) else -1
        # class mapping: 0 -> Big Cats (red), 1 -> Cats (blue)
        if cls == 0:
            color = (0, 0, 200)   # BGR (red-ish)
        elif cls == 1:
            color = (200, 0, 0)   # BGR (blue-ish)
        else:
            color = (200, 200, 200)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness=thickness, lineType=cv2.LINE_AA)
        # IMPORTANT: NO cv2.putText anywhere -> no labels
    # convert back to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

# -----------------------------
# INFERENCE: detection or classification
# -----------------------------
if uploaded is not None:
    # load PIL image
    img_pil = Image.open(uploaded).convert("RGB")

    if st.session_state.mode == "Deteksi Objek":
        if not STATUS["yolo_loaded"]:
            st.info("YOLO not loaded (check badges).")
        else:
            try:
                # Run YOLO (DO NOT call .plot())
                results = yolo_model(img_pil)

                # Try to extract boxes and class ids robustly
                boxes = []
                classes = []
                try:
                    # Preferred access: results[0].boxes.xyxy / .cls
                    r0 = results[0]
                    # .boxes could be an object with attributes
                    if hasattr(r0, "boxes"):
                        b = r0.boxes
                        # xyxy as tensor or numpy
                        if hasattr(b, "xyxy"):
                            boxes = b.xyxy.cpu().numpy() if hasattr(b.xyxy, "cpu") else np.array(b.xyxy)
                        elif hasattr(b, "xyxyn"):
                            boxes = b.xyxyn.cpu().numpy() if hasattr(b.xyxyn, "cpu") else np.array(b.xyxyn)
                        else:
                            boxes = np.array([])

                        # class ids
                        if hasattr(b, "cls"):
                            classes = b.cls.cpu().numpy() if hasattr(b.cls, "cpu") else np.array(b.cls)
                        elif hasattr(b, "cls_id"):
                            classes = b.cls_id.cpu().numpy() if hasattr(b.cls_id, "cpu") else np.array(b.cls_id)
                        else:
                            classes = np.array([])
                    else:
                        boxes = np.array([])
                        classes = np.array([])
                except Exception:
                    boxes = np.array([])
                    classes = np.array([])

                # If boxes empty, try older attributes
                if boxes is None or len(boxes) == 0:
                    # try results[0].masks or results[0].boxes.data
                    try:
                        # some versions store .boxes.data
                        data = results[0].boxes.data if hasattr(results[0].boxes, "data") else None
                        if data is not None:
                            boxes = data[:, :4].cpu().numpy() if hasattr(data, "cpu") else np.array(data)[:, :4]
                            classes = data[:, 5].cpu().numpy() if hasattr(data, "cpu") else np.array(data)[:, 5]
                    except Exception:
                        pass

                # convert to numpy
                boxes = np.array(boxes) if boxes is not None else np.array([])
                classes = np.array(classes) if classes is not None else np.array([])

                # Ensure boxes are in pixel coords (if normalized, scale)
                # Heuristic: if values <=1 -> normalized
                if boxes.size != 0:
                    if boxes.max() <= 1.01:
                        w, h = img_pil.size
                        boxes[:, [0,2]] *= w
                        boxes[:, [1,3]] *= h
                    # clip to image
                    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_pil.size[0]-1)
                    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_pil.size[0]-1)
                    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_pil.size[1]-1)
                    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_pil.size[1]-1)

                # DRAW only boxes (NO text)
                img_result = draw_boxes_only(img_pil, boxes, class_ids=classes)

                # Show only final image (do not show original)
                st.image(img_result, use_column_width=True)

            except Exception as e:
                st.error(f"YOLO inference error: {e}")

    else:
        # Classification mode (0=Big Cats, 1=Cats)
        if not STATUS["clf_loaded"]:
            st.info("Classifier not loaded.")
        else:
            try:
                img_res = img_pil.resize((224, 224))
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

# -----------------------------
# Footer / gallery omitted for brevity
# -----------------------------
