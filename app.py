import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import timedelta
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Setup
st.set_page_config(layout="wide")

# Load data & scalers
@st.cache_data
def load_data():
    df = pd.read_excel("databersih.xlsx")
    # PASTIKAN KODE PRODUK DIBACA SEBAGAI TEKS (STRING) UNTUK MENGHILANGKAN KOMA
    df['KodeProduk'] = df['KodeProduk'].astype(np.int64).astype(str)
    df['WeekStart'] = pd.to_datetime(df['WeekStart'])
    return df

@st.cache_resource
def load_scalers():
    with open("models/feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    with open("models/target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)
    with open("models/selected_features.pkl", "rb") as f:
        selected_features = pickle.load(f)
    return feature_scaler, target_scaler, selected_features

df = load_data()
feature_scaler, target_scaler, selected_features = load_scalers()
df['LabelProduk'] = df.apply(lambda row: f"{row['KodeProduk']} - {row['NamaProduk']}", axis=1)
produk_map = dict(zip(df['LabelProduk'], df['KodeProduk']))

sequence_length = 10

def create_sequences(data, seq_len=10):
    return np.array([data[i:i + seq_len] for i in range(len(data) - seq_len)])

# Sidebar navigasi
if 'page' not in st.session_state:
    st.session_state.page = 'Homepage'

st.sidebar.title("Navigasi")
for menu in ["Homepage", "Korelasi", "Prediksi", "Forecasting", "Strategi"]:
    if st.sidebar.button(menu, use_container_width=True, 
        type="primary" if st.session_state.page == menu else "secondary"):
        st.session_state.page = menu
        st.rerun()

# === HOMEPAGE ===
if st.session_state.page == "Homepage":
    st.title("DASHBOARD PREDIKSI PENJUALAN TOKO 'ANTIANSHOP' ")
    st.write("Selamat datang di Dashboard Prediksi Penjualan Toko 'Antianshop' "
            "Dalam ekosistem Marketplace, prediksi penjualan menjadi faktor penting bagi para penjual dalam menentukan strategi bisnis yang optimal. "
            "Namun, banyak penjual masih menghadapi kesulitan dalam memperkirakan permintaan pasar secara akurat, "
            "sehingga dapat mengakibatkan stok barang yang tidak seimbang serta keputusan pemasaran yang kurang efektif "
            "dengan model prediksi toko 'Antianshop' ini diharapkan dapat membantu pelaku bisnis dalam mengoptimalkan strategi penjualan, "
            "mengetahui faktor yang mempengaruhi penjualan serta mengelola stok produk berdasarkan prediksi penjualannya. ")
   
    st.subheader("Dataset")
    st.info("Data bisa menggunakan data peforma penjualan dari toko shopee lain dengan kolom yang sama") 
    st.write("Kolom yang digunakan yaitu KodeProduk, NamaProduk, Variasi, Kunjungan, HalamanDilihat, TanpaBeli, KlikSearch, Suka, TambahKeranjang, MasukKeranjang, JumlahTerjual, Penjualan, Waktu, Harga, Rating, dan KlikIklan")
    st.write(df)
    st.subheader("Penjelasan Aplikasi")
    st.write("""
    Aplikasi ini menggunakan data historis penjualan untuk memprediksi dan memforecast jumlah terjual produk.
    - Korelasi: Menampilkan fitur paling berpengaruh terhadap JumlahTerjual.
    - Prediksi: Model Bidirectional LSTM per-produk untuk prediksi mingguan.
    - Forecasting: Perkiraan penjualan beberapa minggu ke depan.
    - Strategi: Rekomendasi produk yang perlu diprioritaskan stoknya.
    """)

# === KORELASI ===
elif st.session_state.page == "Korelasi":
    st.title("Korelasi antar Fitur")
    st.write("Korelasi ini bertujuan untuk mengetahui seberapa besar hubungan atau pengaruh "
             "masing-masing fitur terhadap nilai **JumlahTerjual**")
    corr_matrix = df.corr(numeric_only=True)
    jumlah_terjual_corr = corr_matrix['JumlahTerjual'].abs().sort_values(ascending=False)
    st.write(corr_matrix)
    st.info(f"Fitur terpilih (korelasi >= 0.4 dengan JumlahTerjual): {selected_features}")

# === PREDIKSI ===
elif st.session_state.page == "Prediksi":
    st.title("Prediksi Penjualan Mingguan")

    selected_label = st.selectbox("Pilih Produk", list(produk_map.keys()))
    selected_code = produk_map[selected_label]

    df_product = df[df['KodeProduk'] == selected_code].copy()
    if len(df_product) >= 11:
        model_path = f"models/model_{selected_code}.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
            X = feature_scaler.transform(df_product[selected_features])
            X_seq = create_sequences(X, 10)
            y_true = df_product['JumlahTerjual'].values[10:]
            y_pred = target_scaler.inverse_transform(model.predict(X_seq))

            # --- PENAMBAHAN KODE EVALUASI ---
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            st.subheader("Evaluasi Kinerja Model")
            col1, col2 = st.columns(2)
            col1.metric(label="📈 Mean Absolute Error (MAE)", value=f"{mae:.2f}")
            col2.metric(label="🎯 Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")
            

            # --- AKHIR PENAMBAHAN ---
            
            st.subheader("Grafik Perbandingan Prediksi vs Aktual")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(y_true, label='Aktual', marker='o')
            ax.plot(y_pred, label='Prediksi', linestyle='--')
            ax.set_title(f'Kinerja Model untuk Produk: {selected_code}')
            ax.set_xlabel("Minggu")
            ax.set_ylabel("Jumlah Terjual")
            ax.legend()
            st.pyplot(fig)
        else:
            st.error(f"Model untuk produk dengan kode {selected_code} tidak ditemukan.")
    else:
        st.warning("Data historis produk ini tidak cukup untuk melakukan prediksi (kurang dari 11 minggu).")
# === FORECASTING ===
elif st.session_state.page == "Forecasting":
    st.title("Forecasting Mingguan ke Depan")

    selected_label = st.selectbox("Pilih Produk", list(produk_map.keys()), key="forecast")
    selected_code = produk_map[selected_label]
    steps = st.slider("Forecast berapa minggu ke depan?", 1, 12, 4)

    df_product = df[df['KodeProduk'] == selected_code].copy()
    nama_produk = df_product['NamaProduk'].iloc[0] if 'NamaProduk' in df_product.columns else "Nama Tidak Diketahui"
    st.write(f"**Nama Produk:** {nama_produk}")
    if len(df_product) >= 11:
        model_path = f"models/model_{selected_code}.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
            X = feature_scaler.transform(df_product[selected_features])
            current_seq = X[-10:]
            future = []
            date = df_product['WeekStart'].max()
            dates = []
            for _ in range(steps):
                # Prediksi dari model
                y_pred_scaled = model.predict(np.expand_dims(current_seq, axis=0))
                y_pred = target_scaler.inverse_transform(y_pred_scaled)[0][0]
                
                # Simpan hasil forecast ke list
                future.append(int(round(y_pred)))
                
                # Ambil baris terakhir dari sequence sebagai dasar
                new_row = current_seq[-1].copy()
                
                # Ganti nilai fitur JumlahTerjual dengan hasil prediksi yang sudah diskalakan
                y_pred_scaled_for_input = target_scaler.transform([[y_pred]])[0][0]
                new_row[selected_features.index("JumlahTerjual")] = y_pred_scaled_for_input
                
                # Update sequence
                current_seq = np.vstack((current_seq[1:], new_row))
                
                # Tambah tanggal
                date += timedelta(days=7)
                dates.append(date)

            # Buat dataframe hasil forecast
            df_result = pd.DataFrame({"Tanggal": dates, "Forecast": future})
            st.dataframe(df_result)
        else:
            st.error("Model tidak ditemukan.")
    else:
        st.warning("Data belum cukup panjang.")

# === STRATEGI ===
elif st.session_state.page == "Strategi":
    st.title("📦 Strategi Stok Minggu Depan")

    st.info("Berikut adalah 10 produk dengan prediksi penjualan tertinggi minggu depan. Disarankan untuk menyiapkan stok lebih banyak agar tidak kehabisan.")

    result = []

    for kode in df['KodeProduk'].unique():
        df_p = df[df['KodeProduk'] == kode].copy()
        if len(df_p) < 11:
            continue
        model_path = f"models/model_{kode}.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
            X = feature_scaler.transform(df_p[selected_features])
            input_seq = X[-10:]
            y_pred_scaled = model.predict(np.expand_dims(input_seq, axis=0))
            y = target_scaler.inverse_transform(y_pred_scaled)[0][0]
            # Ambil nama produk dari df utama
            nama_produk = df_p['NamaProduk'].iloc[0] if 'NamaProduk' in df_p.columns else "Nama Tidak Diketahui"
            result.append((kode, nama_produk, y))

    # Ambil 10 terbesar
    result = sorted(result, key=lambda x: x[2], reverse=True)[:10]

    for i, (kode, nama, val) in enumerate(result):
       st.markdown(
            f"""
            <div style='font-size:22px; font-weight:bold; margin-top:20px;'>
                {i+1}. {kode} - {nama}
            </div>
            <div style='font-size:18px; margin-bottom:10px;'>
                📈 <b>Prediksi Penjualan:</b> {int(round(val))} produk<br>
                🛒 <b>Rekomendasi:</b> Siapkan stok tambahan minggu depan.
            </div>
            """, 
            unsafe_allow_html=True
        )
