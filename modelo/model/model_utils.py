# model/model_utils.py
import joblib
from config import MODEL_PATH

def save_model(model):
    joblib.dump(model, MODEL_PATH)

def load_model():
    return joblib.load(MODEL_PATH)
