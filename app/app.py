import sys, os

# --------------------------------------------------------------------
# FIX 1 — Prevent GPU/METAL crash on macOS BEFORE importing tensorflow
# --------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_METAL_DISABLE"] = "1"

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------
import streamlit as st
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# --------------------------------------------------------------------
# FIX 2 — Rebuild the SAME MobileNetV2 architecture used during training
#         Then load ONLY the weights from disease_model_final.h5
# --------------------------------------------------------------------
def build_disease_model(input_shape=(224,224,3), num_classes=38):
    base = MobileNetV2(weights=None, include_top=False, input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    out = Dense(num_classes, activation="softmax", name="dense")(x)
    return Model(inputs=base.input, outputs=out)

# Build model and load weights safely
model = build_disease_model(num_classes=38)
WEIGHTS_PATH = "models/disease_model_final.h5"
model.load_weights(WEIGHTS_PATH, by_name=True)
model.trainable = False


# --------------------------------------------------------------------
# Import your custom utilities
# --------------------------------------------------------------------
from utils.leaf_severity import get_leaf_severity
from utils.soil_utils import predict_soil_outputs
from utils.final_severity import final_severity


# --------------------------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------------------------
st.title("🌱 Smart Crop Disease & Soil Intelligence System")

uploaded = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

N = st.number_input("Nitrogen (N)", value=10)
P = st.number_input("Phosphorus (P)", value=5)
K = st.number_input("Potassium (K)", value=40)
pH = st.number_input("Soil pH", value=6.5)
EC = st.number_input("EC", value=1.0)
OC = st.number_input("Organic Carbon", value=1.2)
Moisture = st.number_input("Moisture (%)", value=30)
Temp = st.number_input("Soil Temperature", value=25)

crop = st.selectbox("Crop Type", 
    ["Rice","Wheat","Maize","Tomato","Potato","Cotton","Soybean","Sugarcane"]
)

# --------------------------------------------------------------------
# MAIN ANALYSIS BUTTON
# --------------------------------------------------------------------
if st.button("Analyze"):

    if uploaded is None:
        st.error("Please upload an image.")
        st.stop()

    # ---- IMAGE DISEASE CLASSIFICATION ----
    img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, 1)
    resized = cv2.resize(img, (224,224))

    pred = model.predict(np.expand_dims(resized/255.0, axis=0))[0]
    class_idx = np.argmax(pred)

    st.subheader(f"Disease Class: {int(class_idx)}")

    # ---- LEAF SEVERITY ----
    leaf_sev, mask = get_leaf_severity(img)
    st.write(f"Leaf Severity: **{leaf_sev:.2f}%**")

    # ---- SOIL MODELS (fertility / deficiency / recommendation) ----
    fert_class, def_class, reco_class = predict_soil_outputs(
        N, P, K, pH, EC, OC, Moisture, Temp, crop
    )

    st.write(f"Soil Fertility: **{fert_class}**")
    st.write(f"Nutrient Deficiency: **{def_class}**")
    st.write(f"Fertilizer Recommendation: **{reco_class}**")

    # ---- FINAL SEVERITY ----
    sev = final_severity(leaf_sev, fert_class)
    st.subheader(f"Final Severity: **{sev:.2f}**")

    st.success("Analysis complete!")
