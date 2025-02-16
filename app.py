import streamlit as st
import pandas as pd
from predict_price_modul import price_prediction, model, encoder, scaler

# Mengatur Layout Halaman
st.set_page_config(layout="wide")

st.markdown("""
    <h1 style='text-align: center'>DIAMOND PRICE CALCULATOR 💎</h1>
""", unsafe_allow_html=True)

# Meletakkan gambar di tengah
col1, col2, col3 = st.columns([1,2,1])  # Membagi layout menjadi tiga kolom dengan kolom tengah lebih besar

with col2:  
    st.image("perfect-diamond-isolated-on-shiny-background.jpg", width=300)

# Meletakkan informasi variabel
st.subheader("📜 Diamond Information")
df_info = pd.DataFrame({
    "Variable": ["Carat", "Cut", "Clarity", "Color", "Table", "Depth", "X", "Y", "Z"],
    "Description": [
        "Berat berlian dalam satuan carat.",
        "Kualitas potongan berlian (Fair, Good, Very Good, Premium, Ideal).",
        "Kejernihan berlian dari I1 (terburuk) hingga IF (terjernih).",
        "Warna berlian dari J (kekuningan) hingga D (tidak berwarna).",
        "Lebar bagian atas berlian dalam persen.",
        "Kedalaman berlian dalam persen.",
        "Panjang berlian dalam mm.",
        "Lebar berlian dalam mm.",
        "Tinggi berlian dalam mm."
    ]
})
st.dataframe(df_info)

#Judul sidebar
st.sidebar.header("🔍 Diamond Features")

# Sidebar untuk Input Fitur Berlian
with st.sidebar.expander("Diamond Features", expanded=True):
    carat = st.sidebar.number_input("Carat", min_value=0.10, max_value=7.00, value=1.00, step=0.01, format="%.2f", help="Berat berlian dalam satuan carat.")
    cut = st.sidebar.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=0, help="Kualitas potongan berlian.")
    color = st.sidebar.selectbox("Color", ["J", "I", "H", "G", "F", "E", "D"], index=0, help="Warna berlian.")
    clarity = st.sidebar.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], index=0, help="Kejernihan berlian.")
    depth = st.sidebar.number_input("Depth", min_value=1.00, max_value=100.00, value=50.00, step=0.01, format="%.2f", help="Kedalaman berlian dalam %.")
    table = st.sidebar.number_input("Table", min_value=1.00, max_value=100.00, value=50.00, step=0.01, format="%.2f", help="Lebar bagian atas berlian dalam %.")
    x = st.sidebar.number_input("X (Length)", min_value=1.00, max_value=10.00, value=5.00, step=0.01, format="%.2f", help="Panjang berlian dalam mm.")
    y = st.sidebar.number_input("Y (Width)", min_value=1.00, max_value=10.00, value=5.00, step=0.01, format="%.2f", help="Lebar berlian dalam mm.")
    z = st.sidebar.number_input("Z (Height)", min_value=1.00, max_value=10.00, value=5.00, step=0.01, format="%.2f", help="Tinggi berlian dalam mm.")

#st.sidebar.number_input() untuk variabel numerik
#st.sidebar.selectbox() untuk variabel kategori
#step=0.1 --> mengizinkan perubahan dalam langkah 0.1
#help="..." memberikan penjelasan singkat

# Tampilkan Data Fitur Berlian
diamond_data = pd.DataFrame({
    "carat": [carat],
    "cut": [cut],
    "color": [color],
    "clarity": [clarity],
    "depth": [depth],
    "table": [table],
    "x": [x],
    "y": [y],
    "z": [z]
})

st.subheader("💎 Diamond Feature Data")
# Terapkan format hanya untuk kolom numerik
st.write(diamond_data.style.format({
    "carat": "{:.2f}",
    "depth": "{:.2f}",
    "table": "{:.2f}",
    "x": "{:.2f}",
    "y": "{:.2f}",
    "z": "{:.2f}"
}))

if model is None:
    st.error("Model tidak ditemukan. Pastikan file 'xgbr_model.pkl' ada di direktori yang benar.")
else: #Jika model tersedia, lakukan prediksi harga
    predicted_price = price_prediction(carat, cut, x, y, z, color, depth, clarity, table, model, encoder, scaler)
    st.write(f"### 💰 Estimated Price: **{predicted_price}**")

print("Encoder categories:", encoder.categories_)
