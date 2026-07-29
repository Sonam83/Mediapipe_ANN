import numpy as np
import tensorflow as tf
import joblib

from utils.preprocessing import normalize_landmarks


# -------------------------
# Load Saved Files
# -------------------------

MODEL_PATH = "models/emotion_model.keras"

SCALER_PATH = "models/scaler.pkl"

ENCODER_PATH = "models/label_encoder.pkl"


model = tf.keras.models.load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

encoder = joblib.load(ENCODER_PATH)


# -------------------------
# Prediction Function
# -------------------------

def predict_emotion(landmarks):

    landmarks = normalize_landmarks(landmarks)

    landmarks = landmarks.reshape(1, -1)

    landmarks = scaler.transform(landmarks)

    prediction = model.predict(
        landmarks,
        verbose=0
    )

    emotion_index = np.argmax(prediction)

    emotion = encoder.inverse_transform(
        [emotion_index]
    )[0]

    confidence = float(
        np.max(prediction)
    )

    return emotion, confidence