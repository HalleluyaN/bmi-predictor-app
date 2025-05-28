import os
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 1) Load TFLite feature extractor based on model type
@st.cache_resource
def load_feature_model(model_type: str):
    if model_type == "VGG19":
        model_path = "feature_extractors/vgg19_feature_extractor.tflite"
    else:  # EfficientNet
        model_path = "feature_extractors/efficientnet_feature_extractor.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

# 2) Load scikit-learn regressors
@st.cache_resource
def load_regressors():
    specs = {
        "VGG19-MLP":       "regressors/vgg19_ensemble_model.pkl",
        "EfficientNet-B3": "regressors/Ridge_regressor.pkl"
    }
    loaded = {}
    for name, path in specs.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Regressor not found: {path}")
        loaded[name] = joblib.load(path)
    return loaded

# 3) Image preprocessing (correct input size per model)
def preprocess_image(img: Image.Image, model_type: str):
    if model_type == "VGG19":
        size, fn = (224, 224), vgg_preprocess
    else:
        size, fn = (300, 300), eff_preprocess
    img = img.resize(size).convert("RGB")
    arr = img_to_array(img)[None, ...]
    return fn(arr).astype(np.float32)

# 4) Single-frame BMI prediction
def predict_bmi(img, interpreter, in_det, out_det, regressor, gender_vector, model_type):
    data = preprocess_image(img, model_type)
    interpreter.set_tensor(in_det[0]["index"], data)
    interpreter.invoke()
    feats = interpreter.get_tensor(out_det[0]["index"])  # (1, N)
    expected = getattr(regressor, "n_features_in_", None)
    actual = feats.shape[1]
    if expected is not None:
        if actual + gender_vector.shape[1] == expected:
            feats = np.hstack([feats, gender_vector])
        elif actual != expected:
            raise ValueError(f"Regressor expects {expected} features but got {actual}")
    return float(regressor.predict(feats)[0])

# 5) Live-frame wrapper
def predict_bmi_from_frame(frame, *args):
    img = Image.fromarray(frame)
    return predict_bmi(img, *args)

# 6) VideoTransformer for webcam
class LiveBMI(VideoTransformerBase):
    def __init__(self, model_type, regressor, gender_vector):
        self.interpreter, self.in_det, self.out_det = load_feature_model(model_type)
        self.regressor = regressor
        self.gender_vector = gender_vector
        self.model_type = model_type

    def transform(self, frame):
        import cv2
        img = frame.to_ndarray(format="bgr24")
        try:
            bmi = predict_bmi_from_frame(
                img,
                self.interpreter, self.in_det, self.out_det,
                self.regressor, self.gender_vector,
                self.model_type
            )
            cv2.putText(img, f"BMI: {bmi:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception:
            pass
        return img

# 7) BMI category
def get_bmi_category(bmi: float) -> str:
    if bmi < 18.5: return "Underweight"
    if bmi < 25:   return "Normal weight"
    if bmi < 30:   return "Overweight"
    return "Obese"

# 8) Streamlit UI
st.set_page_config(page_title="BMI Predictor", layout="centered")
st.title("🤳 Predict Your BMI!")

# ——— Gender selector — everything else is hidden until this is set
gender = st.selectbox("Select your gender:", ["", "Male", "Female"])
if gender:
    # map to pipeline
    if gender == "Male":
        model_type, model_key = "VGG19", "VGG19-MLP"
        gender_vector        = np.array([[1]], dtype=np.float32)
    else:
        model_type, model_key = "EfficientNet", "EfficientNet-B3"
        gender_vector        = np.array([[0]], dtype=np.float32)

    # load models
    regressors           = load_regressors()
    regressor            = regressors[model_key]
    interpreter, in_det, out_det = load_feature_model(model_type)

    # show pipeline info
    st.markdown(
        f"**Active pipeline:**  \n"
        f"- Feature extractor: `{model_type}`  \n"
        f"- Regressor model:   `{model_key}`"
    )

    # Input mode
    mode = st.radio(
        "How would you like to provide your image?",
        ["Upload a photo", "Take a photo", "Live webcam"]
    )

    # Live webcam
    if mode == "Live webcam":
        st.write("▶️ Starting live webcam…")
        webrtc_streamer(
            key="live-bmi",
            video_transformer_factory=lambda: LiveBMI(model_type, regressor, gender_vector),
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
        )
    else:
        # file or snapshot branch
        img = None
        if mode == "Upload a photo":
            upl = st.file_uploader("Upload an image (JPG/PNG/BMP)", type=["jpg","jpeg","png","bmp"])
            if upl:
                img = Image.open(upl)
        elif mode == "Take a photo":
            snap = st.camera_input("Take a photo with your camera")
            if snap:
                img = Image.open(snap)

        # once image is available, show and predict
        if img is not None:
            st.image(img, caption="Your input", use_container_width=True)
            if st.button("🔍 Predict BMI"):
                bmi = predict_bmi(
                    img,
                    interpreter, in_det, out_det,
                    regressor, gender_vector,
                    model_type
                )
                cat = get_bmi_category(bmi)
                st.success(f"📏 Predicted BMI: {bmi:.1f}")
                st.info(f"Category: {cat}")
