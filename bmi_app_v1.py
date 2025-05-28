import os
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ——————————————————————————————————————————————————————————————
# 1) Load TFLite feature extractor
# ——————————————————————————————————————————————————————————————
@st.cache_resource
def load_feature_model():
    tflite_path = os.path.join("feature_extractors", "vgg19_feature_extractor.tflite")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

# ——————————————————————————————————————————————————————————————
# 2) Load scikit‐learn regressors
# ——————————————————————————————————————————————————————————————
@st.cache_resource
def load_regressors():
    base = "regressors"
    files = {
        "VGG19-MLP":      "vgg19_mlp_model.pkl",
        "VGG19-Ensemble":"vgg19_ensemble_model.pkl"
    }
    return {
        name: joblib.load(os.path.join(base, fname))
        for name, fname in files.items()
    }

# ——————————————————————————————————————————————————————————————
# 3) Image preprocessing helper
# ——————————————————————————————————————————————————————————————
def preprocess_image(img: Image.Image):
    img = img.resize((224, 224)).convert("RGB")
    arr = img_to_array(img)[None, ...]
    return preprocess_input(arr).astype(np.float32)

# ——————————————————————————————————————————————————————————————
# 4) BMI prediction helper (single‐frame)
# ——————————————————————————————————————————————————————————————
def predict_bmi(img: Image.Image, interpreter, in_det, out_det, regressor):
    data = preprocess_image(img)
    interpreter.set_tensor(in_det[0]["index"], data)
    interpreter.invoke()
    feats = interpreter.get_tensor(out_det[0]["index"])
    return float(regressor.predict(feats)[0])

# ——————————————————————————————————————————————————————————————
# 5) BMI prediction helper (per‐frame for live)
# ——————————————————————————————————————————————————————————————
def predict_bmi_from_frame(frame, interpreter, in_det, out_det, regressor):
    img = Image.fromarray(frame)
    return predict_bmi(img, interpreter, in_det, out_det, regressor)

# ——————————————————————————————————————————————————————————————
# 6) Transformer for live video
# ——————————————————————————————————————————————————————————————
class LiveBMI(VideoTransformerBase):
    def __init__(self, regressor):
        self.interpreter, self.in_det, self.out_det = load_feature_model()
        self.regressor = regressor

    def transform(self, frame):
        import cv2
        img = frame.to_ndarray(format="bgr24")
        try:
            bmi = predict_bmi_from_frame(
                img, self.interpreter, self.in_det, self.out_det, self.regressor
            )
            text = f"BMI: {bmi:.1f}"
            cv2.putText(img, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception:
            pass
        return img

# ——————————————————————————————————————————————————————————————
# 7) BMI category helper
# ——————————————————————————————————————————————————————————————
def get_bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# ——————————————————————————————————————————————————————————————
# 8) Streamlit UI
# ——————————————————————————————————————————————————————————————
st.set_page_config(page_title="BMI Predictor", layout="centered")
st.title("🤳 Predict Your BMI!")

# Load feature extractor and regressors
interpreter, in_det, out_det = load_feature_model()
regressors = load_regressors()

# Model selector
model_name = st.selectbox("Choose regression model:", list(regressors.keys()))
regressor = regressors[model_name]

# Input mode selector with three distinct options
mode = st.radio(
    "How would you like to provide your image?",
    ["Upload a file", "Take snapshot", "Live webcam"]
)

# Live‐webcam path
if mode == "Live webcam":
    st.write("▶️ Starting live webcam…")
    webrtc_streamer(
        key="live-bmi",
        video_transformer_factory=lambda: LiveBMI(regressor),
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    st.stop()

# Upload a file
img = None
if mode == "Upload a file":
    uploaded = st.file_uploader("Upload an image (JPG/PNG/BMP)", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded:
        img = Image.open(uploaded)

# Take snapshot
elif mode == "Take snapshot":
    snapped = st.camera_input("Take a photo with your camera")
    if snapped:
        img = Image.open(snapped)

# If we have an image, run prediction
if img:
    st.image(img, caption="Your input", use_column_width=True)
    if st.button("🔍 Predict BMI"):
        bmi = predict_bmi(img, interpreter, in_det, out_det, regressor)
        cat = get_bmi_category(bmi)
        st.success(f"📏 Predicted BMI: {bmi:.1f}")
        st.info(f"Category: {cat}")
