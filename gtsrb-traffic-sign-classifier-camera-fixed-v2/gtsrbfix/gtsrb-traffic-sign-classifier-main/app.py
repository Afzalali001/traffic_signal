import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
import time
import base64
import html

# ============================================================
# TRAFFIC SIGN AI - ADVANCED DASHBOARD
# ============================================================
st.set_page_config(
    page_title="Traffic Sign AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------
# CLASS LABELS
# -----------------------
class_labels = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)", 3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)", 9: "No passing",
    10: "No passing for vehicles > 3.5 tons",
    11: "Right-of-way at intersection", 12: "Priority road",
    13: "Yield", 14: "Stop", 15: "No vehicles",
    16: "Vehicles > 3.5 tons prohibited", 17: "No entry",
    18: "General caution", 19: "Dangerous curve left",
    20: "Dangerous curve right", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road", 24: "Road narrows",
    25: "Road work", 26: "Traffic signals", 27: "Pedestrians",
    28: "Children crossing", 29: "Bicycles crossing", 30: "Ice/snow",
    31: "Wild animals crossing", 32: "End of all limits",
    33: "Turn right ahead", 34: "Turn left ahead", 35: "Ahead only",
    36: "Straight or right", 37: "Straight or left", 38: "Keep right",
    39: "Keep left", 40: "Roundabout mandatory", 41: "End of no passing",
    42: "End of no passing vehicles > 3.5 tons"
}

# -----------------------
# MODEL
# -----------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("traffic_sign_model.keras")

try:
    model = load_model()
    model_status = True
except Exception as e:
    model = None
    model_status = False
    st.error("Model could not be loaded. Check that traffic_sign_model.keras is beside app.py.")
    st.exception(e)

# -----------------------
# SESSION STATE
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_alert" not in st.session_state:
    st.session_state.last_alert = ""

# -----------------------
# HELPERS
# -----------------------
def preprocess(img):
    """Match the original model pipeline: BGR/RGB image -> 32x32 -> [0,1]."""
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def predict(img):
    batch = preprocess(img)
    preds = model.predict(batch, verbose=0)[0]
    pred_class = int(np.argmax(preds))
    confidence = float(preds[pred_class])
    return pred_class, confidence, preds

def center_crop(img, ratio=0.72):
    """Fallback crop for a centered sign."""
    h, w = img.shape[:2]
    side = int(min(h, w) * ratio)
    y1 = max(0, (h - side) // 2)
    x1 = max(0, (w - side) // 2)
    return img[y1:y1+side, x1:x1+side]


def square_crop(img, bbox, pad=0.30):
    """Turn a bounding box into a padded square crop."""
    h, w = img.shape[:2]
    x, y, bw, bh = [int(v) for v in bbox]
    side = max(bw, bh)
    cx, cy = x + bw / 2.0, y + bh / 2.0
    side = int(side * (1.0 + pad))
    x1 = max(0, int(cx - side / 2))
    y1 = max(0, int(cy - side / 2))
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    # Re-center after clipping at image borders.
    x1 = max(0, x2 - side)
    y1 = max(0, y2 - side)
    crop = img[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def find_sign_candidates(img):
    """Find likely traffic-sign regions using color/shape cues.

    This is a lightweight localizer for the existing classifier. It does not
    retrain or replace the GTSRB CNN. It generates several candidate crops and
    lets the classifier score them.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red, blue and yellow are common dominant sign colors in GTSRB.
    red1 = cv2.inRange(hsv, (0, 70, 55), (12, 255, 255))
    red2 = cv2.inRange(hsv, (165, 70, 55), (179, 255, 255))
    blue = cv2.inRange(hsv, (90, 60, 45), (140, 255, 255))
    yellow = cv2.inRange(hsv, (15, 65, 65), (40, 255, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), cv2.bitwise_or(blue, yellow))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    image_area = float(h * w)

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < image_area * 0.006 or area > image_area * 0.55:
            continue
        if bw < 28 or bh < 28:
            continue
        ratio = bw / float(bh)
        if ratio < 0.45 or ratio > 2.2:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True) if peri else []
        fill = cv2.contourArea(c) / float(max(1, area))
        center_dist = ((x + bw/2 - w/2) / w) ** 2 + ((y + bh/2 - h/2) / h) ** 2
        # Prefer sign-like filled regions and avoid tiny/edge noise.
        shape_score = min(1.0, fill / 0.45) if fill else 0.0
        center_score = max(0.0, 1.0 - 2.2 * center_dist)
        poly_score = 1.0 if len(approx) in (3, 4, 6, 8) else 0.7
        geom_score = 0.45 * shape_score + 0.35 * center_score + 0.20 * poly_score
        crop, rect = square_crop(img, (x, y, bw, bh), pad=0.35)
        if crop.size:
            candidates.append((geom_score, crop, rect))

    candidates.sort(key=lambda z: z[0], reverse=True)

    # Add a center fallback so signs with weak color segmentation still work.
    fallback = center_crop(img, 0.72)
    fh, fw = fallback.shape[:2]
    fx = max(0, (w - fw) // 2)
    fy = max(0, (h - fh) // 2)
    candidates.append((0.50, fallback, (fx, fy, fx + fw, fy + fh)))

    # Keep only the best diverse candidates.
    selected = []
    for item in candidates:
        _, _, rect = item
        x1, y1, x2, y2 = rect
        duplicate = False
        for _, _, r2 in selected:
            ax1, ay1, ax2, ay2 = r2
            ix1, iy1, ix2, iy2 = max(x1, ax1), max(y1, ay1), min(x2, ax2), min(y2, ay2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            a1 = max(1, (x2-x1)*(y2-y1)); a2 = max(1, (ax2-ax1)*(ay2-ay1))
            if inter / float(a1 + a2 - inter) > 0.65:
                duplicate = True
                break
        if not duplicate:
            selected.append(item)
        if len(selected) >= 6:
            break
    return selected


def predict_camera(img, min_confidence=0.60, min_margin=0.10):
    """Localize likely sign regions, classify each, and reject ambiguous results."""
    candidates = find_sign_candidates(img)
    scored = []
    for geom_score, crop, rect in candidates:
        pred_class, confidence, preds = predict(crop)
        top2 = np.sort(preds)[-2:] if len(preds) >= 2 else np.array([0.0, confidence])
        margin = float(top2[-1] - top2[-2])
        # Candidate geometry is a tie-breaker, not a replacement for model confidence.
        score = float(confidence) + 0.08 * float(geom_score)
        scored.append((score, confidence, margin, pred_class, preds, crop, rect, geom_score))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0]
    _, confidence, margin, pred_class, preds, crop, rect, geom_score = best
    accepted = confidence >= min_confidence and margin >= min_margin
    return accepted, pred_class, confidence, margin, preds, crop, rect, geom_score, len(candidates)

def speed_limit(label):
    import re
    m = re.search(r"Speed limit \((\d+)km/h\)", label)
    return int(m.group(1)) if m else None

def voice_text(label):
    s = speed_limit(label)
    if s is not None:
        return f"Speed limit is {s} kilometers per hour."
    spoken = label.replace(">", " greater than ").replace("km/h", " kilometers per hour")
    return f"Traffic sign detected: {spoken}."

def add_history(label, confidence, source):
    st.session_state.history.append({
        "Time": time.strftime("%H:%M:%S"),
        "Source": source,
        "Sign": label,
        "Confidence": round(confidence * 100, 2),
    })
    st.session_state.history = st.session_state.history[-50:]

def speak(text_to_speak):
    # Browser-side Web Speech API. No paid API/key required.
    safe = html.escape(text_to_speak).replace("\\", "\\\\").replace("'", "\\'")
    component = f"""
    <script>
    const msg = new SpeechSynthesisUtterance('{safe}');
    msg.lang = 'en-US';
    msg.rate = 0.95;
    msg.pitch = 1.0;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(msg);
    </script>
    """
    st.components.v1.html(component, height=0)

def annotate(img, label, confidence):
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (15, 15), (min(w-15, 650), 105), (20, 20, 20), -1)
    cv2.putText(out, label[:55], (30, 52), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"Confidence: {confidence*100:.1f}%", (30, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (80, 220, 120), 2, cv2.LINE_AA)
    return out

def result_panel(pred_class, confidence, source, speak_enabled=True):
    label = class_labels[pred_class]
    s = speed_limit(label)

    c1, c2, c3 = st.columns(3)
    c1.metric("Detected Sign", label)
    c2.metric("Confidence", f"{confidence*100:.2f}%")
    c3.metric("Source", source)

    if confidence < 0.60:
        st.warning("Low confidence. Keep the sign clear, centered and well lit.")
    elif confidence >= 0.90:
        st.success("High-confidence prediction.")

    if s is not None:
        st.warning(f"🚦 SPEED ALERT: detected speed limit is {s} km/h")

    if speak_enabled:
        if st.button("🔊 Speak Detection", key=f"speak_{source}_{time.time_ns()}"):
            speak(voice_text(label))

# -----------------------
# HEADER
# -----------------------
st.title("🚦 Traffic Sign Recognition AI")
st.caption("CNN-powered GTSRB classifier • Image + Camera + Voice + Analytics")

with st.sidebar:
    st.header("⚙️ Controls")
    confidence_threshold = st.slider(
        "Minimum confidence for alerts", 0.0, 1.0, 0.60, 0.05
    )
    auto_voice = st.checkbox("Auto voice after camera prediction", value=True)
    crop_camera = st.checkbox(
        "Center-crop camera image before prediction",
        value=True,
        help="Recommended: the CNN is a classifier, not an object detector. Keep the sign centered."
    )
    crop_ratio = st.slider(
        "Camera crop size", 0.45, 0.95, 0.68, 0.01,
        help="Smaller values remove more background. Start around 0.68 and adjust so the whole sign stays visible."
    )
    st.divider()
    if model_status:
        st.success("Model loaded")
    else:
        st.error("Model unavailable")
    st.write("Classes:", len(class_labels))
    st.write("Input:", "32 × 32 × 3")

if not model_status:
    st.stop()

# -----------------------
# MAIN TABS
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload", "📷 Camera", "📊 Dashboard", "ℹ️ About"
])

with tab1:
    st.subheader("Upload Traffic Sign Image")
    uploaded_file = st.file_uploader(
        "Choose JPG / JPEG / PNG",
        type=["jpg", "jpeg", "png"],
        key="upload"
    )

    if uploaded_file:
        file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Could not read this image.")
        else:
            pred_class, confidence, preds = predict(img)
            label = class_labels[pred_class]
            add_history(label, confidence, "Upload")

            left, right = st.columns([1.15, 1])
            with left:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         caption="Input image", use_container_width=True)
            with right:
                result_panel(pred_class, confidence, "Upload", speak_enabled=True)

            st.subheader("Top 5 Predictions")
            top5_idx = np.argsort(preds)[-5:][::-1]
            df = pd.DataFrame({
                "Class": [class_labels[int(i)] for i in top5_idx],
                "Confidence": [float(preds[i]) for i in top5_idx],
            }).set_index("Class")
            st.bar_chart(df)

with tab2:
    st.subheader("📷 Camera Traffic Sign Recognition")
    st.info(
        "The camera now searches for likely red/blue/yellow sign regions first, "
        "then sends the best candidate crop to the existing GTSRB CNN. Keep one sign "
        "visible and reasonably large for reliable classification."
    )

    detector_mode = st.selectbox(
        "Camera detection mode",
        ["Auto-locate sign (recommended)", "Centered sign"],
        index=0,
    )
    camera_image = st.camera_input("Capture traffic sign", key="traffic_camera")

    if camera_image:
        file_bytes = np.frombuffer(camera_image.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Camera frame could not be decoded.")
        else:
            if detector_mode == "Auto-locate sign (recommended)":
                accepted, pred_class, confidence, margin, preds, prediction_img, rect, geom_score, n_candidates = predict_camera(
                    img, min_confidence=confidence_threshold, min_margin=0.10
                )
            else:
                prediction_img = center_crop(img, crop_ratio)
                pred_class, confidence, preds = predict(prediction_img)
                top2 = np.sort(preds)[-2:]
                margin = float(top2[-1] - top2[-2]) if len(top2) >= 2 else confidence
                accepted = confidence >= confidence_threshold and margin >= 0.10
                h, w = img.shape[:2]
                side = prediction_img.shape[0]
                rect = ((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2)
                geom_score, n_candidates = 0.5, 1

            label = class_labels[pred_class]

            left, right = st.columns(2)
            with left:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         caption="Original camera frame", use_container_width=True)
            with right:
                st.image(cv2.cvtColor(prediction_img, cv2.COLOR_BGR2RGB),
                         caption="Sign crop sent to CNN", use_container_width=True)

            st.caption(
                f"Candidate regions checked: {n_candidates} • "
                f"Top-1 confidence: {confidence*100:.1f}% • "
                f"Top-1/Top-2 margin: {margin*100:.1f}%"
            )

            if accepted:
                add_history(label, confidence, "Camera")
                annotated = img.copy()
                x1, y1, x2, y2 = [int(v) for v in rect]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 3)
                cv2.rectangle(annotated, (15, 15), (min(img.shape[1]-15, 720), 115), (20, 20, 20), -1)
                cv2.putText(annotated, label[:58], (30, 55), cv2.FONT_HERSHEY_SIMPLEX,
                            0.72, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated, f"Confidence: {confidence*100:.1f}%  Margin: {margin*100:.1f}%",
                            (30, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                            (80, 220, 120), 2, cv2.LINE_AA)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         caption="Detected traffic sign", use_container_width=True)
                result_panel(pred_class, confidence, "Camera", speak_enabled=False)

                if auto_voice:
                    spoken = voice_text(label)
                    speak(spoken)
                    st.success(f"🔊 Voice alert: “{spoken}”")
            else:
                st.warning(
                    f"⚠️ Uncertain/ambiguous detection: {label} ({confidence*100:.1f}%, "
                    f"margin {margin*100:.1f}%). Voice alert was blocked. "
                    "Move the sign closer, keep it fully visible, improve lighting, "
                    "and capture again."
                )

            st.subheader("Top 5 Camera Predictions")
            top5_idx = np.argsort(preds)[-5:][::-1]
            cam_df = pd.DataFrame({
                "Class": [class_labels[int(i)] for i in top5_idx],
                "Confidence": [float(preds[i]) for i in top5_idx],
            }).set_index("Class")
            st.bar_chart(cam_df)

with tab3:
    st.subheader("📊 Detection Dashboard")

    history = pd.DataFrame(st.session_state.history)

    if history.empty:
        st.info("No detections yet. Use Upload or Camera first.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Total Detections", len(history))
        b.metric("Average Confidence", f"{history['Confidence'].mean():.1f}%")
        c.metric("High Confidence", int((history["Confidence"] >= 90).sum()))
        d.metric("Unique Signs", history["Sign"].nunique())

        st.subheader("Detection History")
        st.dataframe(history, use_container_width=True, hide_index=True)

        st.subheader("Confidence by Detection")
        chart_df = history[["Time", "Confidence"]].copy()
        chart_df["Confidence"] = chart_df["Confidence"].astype(float)
        st.line_chart(chart_df.set_index("Time"))

        st.subheader("Most Detected Signs")
        counts = history["Sign"].value_counts()
        st.bar_chart(counts)

        csv = history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Detection History CSV",
            data=csv,
            file_name="traffic_sign_detection_history.csv",
            mime="text/csv",
        )

        if st.button("🗑️ Clear Detection History"):
            st.session_state.history = []
            st.rerun()

with tab4:
    st.subheader("ℹ️ Project Features")
    st.markdown("""
### Current system

- 🧠 **CNN / Keras traffic-sign classifier**
- 📤 **Image upload prediction**
- 📷 **Browser camera capture**
- 🎯 **43 GTSRB classes**
- 📈 **Confidence score**
- 🔝 **Top-5 predictions**
- 🔊 **Browser voice announcement**
- 🚦 **Automatic speed-limit alert**
- 📊 **Detection history dashboard**
- 📥 **CSV export**
- ⚠️ **Low-confidence warning**
- 🎥 **Annotated camera result**

### Example

`Camera → Speed Limit 50 → Confidence 96.4%`

Then the browser can speak:

> Speed limit is 50 kilometers per hour.

### Camera accuracy / important limitation

This project uses a **traffic-sign classifier**, not a complete object detector. The CNN expects a sign-like image resized to 32×32. For camera use, keep one sign centered and reasonably large in the frame; the app now center-crops the frame by default and shows exactly what is sent to the CNN.

Low-confidence camera predictions are treated as **uncertain** and are not spoken, which prevents the system from confidently announcing an obvious wrong guess. For automatic detection of small signs anywhere in a road scene, the next upgrade is a dedicated object-detection/localization model (for example YOLO) followed by this classifier.
""")

st.divider()
st.caption("Traffic Sign AI • Existing trained model preserved • No GTSRB retraining required")
