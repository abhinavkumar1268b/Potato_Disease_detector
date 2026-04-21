"""
app.py — Potato Disease Classifier (v2)
========================================
4-Class Streamlit app:
  • Early Blight
  • Late Blight
  • Healthy
  • Not Potato Leaf  ← new: model rejects non-leaf uploads

Deployment: Streamlit Community Cloud compatible.
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ─── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Potato Disease Classifier",
    page_icon="🥔",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Constants ────────────────────────────────────────────────────────────────

# Class names MUST match alphabetical order of the dataset subfolders
# TF reads: Not_Potato_Leaf, Potato___Early_blight, Potato___Late_blight, Potato___healthy
CLASS_NAMES = [
    "Not Potato Leaf",   # index 0 → Not_Potato_Leaf/
    "Early Blight",      # index 1 → Potato___Early_blight/
    "Late Blight",       # index 2 → Potato___Late_blight/
    "Healthy",           # index 3 → Potato___healthy/
]

# Minimum confidence required to trust the prediction
CONFIDENCE_THRESHOLD = 70.0

# Path to the model file (same directory as app.py)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "potato_disease_model_v2.keras")

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Dark background ───────────────────────────────── */
    .stApp, .stMain, section[data-testid="stSidebar"] {
        background-color: #0E1117;
    }

    /* ── All generic text → white ──────────────────────── */
    html, body, [class*="css"],
    .stMarkdown, .stMarkdown p,
    .stText, label, .stFileUploaderLabel,
    .stProgress > div, .stExpander p,
    div[data-testid="stFileUploaderDropzoneInstructions"] * {
        color: #FFFFFF !important;
    }

    /* ── Title ──────────────────────────────────────────── */
    h1, h2, h3 { color: #4FC3F7 !important; }

    /* ── Dividers ───────────────────────────────────────── */
    hr { border-color: #2A2D36 !important; }

    /* ── File uploader box ──────────────────────────────── */
    [data-testid="stFileUploader"] {
        background-color: #1A1D27;
        border: 1.5px dashed #4FC3F7;
        border-radius: 10px;
    }

    /* ── Expander ───────────────────────────────────────── */
    [data-testid="stExpander"] {
        background-color: #1A1D27;
        border: 1px solid #2A2D36;
        border-radius: 10px;
    }
    [data-testid="stExpander"] summary span {
        color: #FFFFFF !important;
    }

    /* ── Spinner text ───────────────────────────────────── */
    [data-testid="stSpinner"] p { color: #FFFFFF !important; }

    /* ── Result cards ───────────────────────────────────── */
    .result-box {
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 16px;
        font-size: 1.05rem;
        color: #FFFFFF;
    }
    /* Healthy  — dark green tint */
    .result-healthy   { background: #0D2318; border-left: 6px solid #4CAF50; }
    /* Disease  — dark red tint */
    .result-disease   { background: #2A0A0A; border-left: 6px solid #EF5350; }
    /* Invalid  — dark amber tint */
    .result-invalid   { background: #2A1F00; border-left: 6px solid #FFC107; }
    /* Uncertain — dark blue tint */
    .result-uncertain { background: #0A1929; border-left: 6px solid #42A5F5; }

    /* ── Confidence label ───────────────────────────────── */
    .conf-label { font-size: 0.9rem; color: #AAAAAA; margin-bottom: 4px; }

    /* ── Button ─────────────────────────────────────────── */
    [data-testid="stBaseButton-primary"] {
        background-color: #1565C0 !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    [data-testid="stBaseButton-primary"]:hover {
        background-color: #1E88E5 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Model Loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """Load and cache the Keras model. Runs only once per session."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"❌ Model file not found at: `{MODEL_PATH}`\n\n"
            "Please run `python train.py` first, then copy "
            "`potato_disease_model_v2.keras` into the `app/` folder."
        )
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a model-ready batch tensor.
    The model's internal Rescaling layer handles normalisation,
    so we only need to resize and expand dims here.
    """
    image    = image.convert("RGB")             # Ensure 3-channel RGB
    image    = image.resize((256, 256))         # Resize to model input size
    arr      = np.array(image, dtype=np.float32)
    arr      = np.expand_dims(arr, axis=0)      # Shape: (1, 256, 256, 3)
    return arr

# ─── UI — Header ──────────────────────────────────────────────────────────────

st.title("🥔 Potato Disease Classifier")
st.markdown(
    "Upload a **clear photo of a potato leaf** and the AI will classify it as "
    "**Early Blight**, **Late Blight**, or **Healthy**. "
    "Non-leaf images are automatically rejected."
)
st.markdown("---")

# ─── UI — File Uploader ───────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "📸 Upload a potato leaf image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

# ─── UI — Prediction ──────────────────────────────────────────────────────────

if uploaded_file is not None:

    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown("---")

    # Predict button
    if st.button("🔍 Classify Leaf", use_container_width=True, type="primary"):

        with st.spinner("Analysing image…"):
            input_tensor = preprocess_image(image)
            predictions  = model.predict(input_tensor, verbose=0)[0]  # shape: (4,)

        predicted_idx   = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = float(np.max(predictions) * 100)

        # ── Confidence bar ────────────────────────────────────────────────────
        st.markdown(f"<div class='conf-label'>Model confidence</div>",
                    unsafe_allow_html=True)
        st.progress(int(confidence), text=f"{confidence:.1f}%")

        # ── All class probabilities (expander) ────────────────────────────────
        with st.expander("📊 See all class probabilities"):
            for cls, prob in zip(CLASS_NAMES, predictions):
                st.write(f"**{cls}:** {prob * 100:.2f}%")
                st.progress(int(prob * 100))

        st.markdown("---")

        # ── Decision logic ────────────────────────────────────────────────────

        # Case 1: Model is not confident enough
        if confidence < CONFIDENCE_THRESHOLD:
            st.markdown("""
            <div class='result-box result-uncertain'>
                🤔 <strong>Uncertain Prediction</strong><br><br>
                The model is not confident enough about this image.<br>
                Please try uploading a clearer, well-lit photo of the potato leaf.
            </div>""", unsafe_allow_html=True)

        # Case 2: Not a potato leaf
        elif predicted_class == "Not Potato Leaf":
            st.markdown("""
            <div class='result-box result-invalid'>
                ⚠️ <strong>Not a Potato Leaf</strong><br><br>
                Please upload a <strong>clear photo of a potato leaf</strong>.<br>
                The image you uploaded does not appear to be a potato leaf.
            </div>""", unsafe_allow_html=True)

        # Case 3: Healthy leaf
        elif predicted_class == "Healthy":
            st.markdown(f"""
            <div class='result-box result-healthy'>
                ✅ <strong>No Disease Detected</strong><br><br>
                The leaf appears to be <strong>healthy</strong>!<br>
                <em>Confidence: {confidence:.1f}%</em>
            </div>""", unsafe_allow_html=True)

        # Case 4: Disease detected
        else:
            st.markdown(f"""
            <div class='result-box result-disease'>
                🔴 <strong>Disease Detected: {predicted_class}</strong><br><br>
                {"<em>Early Blight</em> is caused by the fungus <em>Alternaria solani</em>. "
                 "It causes dark brown spots with yellow rings on leaves."
                 if predicted_class == "Early Blight" else
                 "<em>Late Blight</em> is caused by <em>Phytophthora infestans</em>. "
                 "It causes water-soaked, dark lesions that spread rapidly."}<br><br>
                <em>Confidence: {confidence:.1f}%</em>
            </div>""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#777777; font-size:0.85rem;'>"
    "Potato Disease Classifier · Built with TensorFlow &amp; Streamlit · "
    "Student Portfolio Project"
    "</div>",
    unsafe_allow_html=True
)