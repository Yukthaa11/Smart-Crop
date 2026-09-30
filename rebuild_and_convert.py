# rebuild_and_convert.py
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
import os
import sys

WEIGHTS_PATH = "models/disease_model.h5"
OUT_PATH = "models/disease_model_final.h5"
num_classes = 38

def build_model(input_shape=(224,224,3), num_classes=38):
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=out)
    return model

print("Building model...")
model = build_model()

print("Loading weights...")
model.load_weights(WEIGHTS_PATH, by_name=True)
print("Weights loaded!")

print("Saving final FULL model (H5 format)...")
model.save(OUT_PATH, save_format="h5")
print("SUCCESS! Saved:", OUT_PATH)
