import streamlit as st
import numpy as np
import sklearn
from sklearn.preprocessing import OrdinalEncoder

# import ml package
import os
import joblib

def load_model(model_file):
    try:
        # Cek file model ada
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file '{model_file}' tidak ditemukan.")

        # Muat model
        loaded_model = joblib.load(model_file)
        return loaded_model
    except ModuleNotFoundError as e:
        st.error("Error: Modul yang diperlukan tidak ditemukan. Pastikan 'xgboost' telah terinstal.")
        st.error(str(e))
        return None
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# Fungsi untuk memuat scaler
def load_scaler(scaler_file):
    try:
        loaded_scaler = joblib.load(scaler_file)
        return loaded_scaler
    except Exception as e:
        st.error(f"Error saat memuat scaler: {e}")
        return None


# muat model
model = load_model("xgbr_model.pkl")
scaler = load_scaler("scaler.pkl")

# Inisialisasi encoder --> terburuk ke terbaik
encoder = OrdinalEncoder(categories=[
    ["Fair", "Good", "Very Good", "Premium", "Ideal"],  # Cut
    ["J", "I", "H", "G", "F", "E", "D"],  # Color
    ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]  # Clarity
])

# Fit encoder --> menyamakan kategori 
encoder.fit([
    ["Fair", "D", "I1"],
    ["Good", "E", "SI2"],
    ["Very Good", "F", "SI1"],
    ["Premium", "G", "VS2"],
    ["Ideal", "H", "VS1"],
    ["Fair", "I", "VVS2"],
    ["Good", "J", "VVS1"],
    ["Very Good", "D", "IF"]
])

# Fungsi prediksi harga diamond
def price_prediction(carat, cut, x, y, z, color, depth, clarity, table, model, encoder, scaler):
    if model is None:
        return "Model not loaded. Check the file path."
    
    try:
        # Encoding kategori
        encoded_features = encoder.transform([[cut, color, clarity]])
        n_cut, n_color, n_clarity = encoded_features[0]

         # Buat array input model
        X = np.array([[carat, n_cut, n_color, n_clarity, depth, table, x, y, z]])
        X_scaled = scaler.transform(X)

        # Prediksi harga
        prediction = model.predict(X_scaled)
        return f"${prediction[0]:,.2f}"
    
    except Exception as e:
        return f"Error dalam prediksi: {e}"

