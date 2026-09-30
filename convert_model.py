from tensorflow.keras.models import load_model

print("Loading H5 model...")
model = load_model("models/disease_model.h5", compile=False)

print("Saving as .keras format...")
model.save("models/disease_model.keras")

print("Conversion complete!")
