import joblib
import numpy as np

fertility_model = joblib.load("models/soil_fertility_model.pkl")
def_model = joblib.load("models/soil_deficiency_model.pkl")
fert_rec_model = joblib.load("models/fertilizer_ml_model.pkl")

le_crop = joblib.load("models/crop_label_encoder.pkl")
le_fert = joblib.load("models/fertility_label_encoder.pkl")
le_def = joblib.load("models/def_label_encoder.pkl")
le_reco = joblib.load("models/reco_label_encoder.pkl")

def predict_soil_outputs(N, P, K, pH, EC, OC, Moisture, Temp, crop):
    crop_enc = le_crop.transform([crop])[0]

    X = np.array([[N,P,K,pH,EC,OC,Moisture,Temp,crop_enc]])

    fert_class = le_fert.inverse_transform(fertility_model.predict(X))[0]
    def_class  = le_def.inverse_transform(def_model.predict(X))[0]
    reco_class = le_reco.inverse_transform(fert_rec_model.predict(X))[0]

    return fert_class, def_class, reco_class
