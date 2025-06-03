import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import io

# Fungsi untuk memuat dan membersihkan data
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error saat membaca file CSV: {e}")
        return None

    # Hapus kolom 'index' jika ada, karena tidak relevan untuk analisis
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    
    # Kolom yang perlu dikonversi ke numerik
    numeric_cols = ['Movies Released', 'Gross', 'Tickets Sold', 
                    'Inflation-Adjusted Gross', 'Top Movie Gross (That Year)', 
                    'Top Movie Inflation-Adjusted Gross (That Year)']
    
    for col in numeric_cols:
        if col in df.columns:
            # Hapus koma dan konversi ke numerik, paksa error menjadi NaN
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # Buat kolom dummy jika tidak ada untuk menghindari error lebih lanjut, isi dengan 0
            df[col] = 0 

    # Hapus baris dengan NaN di kolom numerik penting setelah konversi
    # Fokus pada kolom yang akan digunakan untuk prediksi
    df.dropna(subset=['Year', 'Genre', 'Inflation-Adjusted Gross', 'Tickets Sold', 'Movies Released'], inplace=True)
    
    if 'Year' in df.columns:
        df['Year'] = df['Year'].astype(int)

    return df

# Fungsi untuk membuat sekuens untuk LSTM
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Fungsi untuk melatih model LSTM dan membuat prediksi
# Menggunakan st.cache_data agar model tidak dilatih ulang setiap interaksi kecil jika input sama
@st.cache_data(show_spinner=False, persist="disk") # Menggunakan persist untuk menyimpan cache antar sesi jika memungkinkan
def train_and_predict_lstm(_genre_data, metric_column, n_steps=3, n_features=1, epochs=50, future_predictions=3):
    if _genre_data.empty or len(_genre_data) < n_steps + 1:
        return None, None, "Data tidak cukup untuk genre ini."

    # Normalisasi data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(_genre_data[[metric_column]].values)

    X, y_seq = create_sequences(scaled_data, n_steps)
    if X.shape[0] == 0: # Tidak ada cukup data untuk membuat sekuens
         return None, None, "Data tidak cukup untuk membuat sekuens pelatihan."


    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # Definisikan model LSTM
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Latih model
    model.fit(X, y_seq, epochs=epochs, verbose=0)

    # Buat prediksi untuk masa depan
    temp_input = list(scaled_data[-n_steps:].flatten())
    lst_output = []
    i = 0
    while(i < future_predictions):
        if(len(temp_input) > n_steps):
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape((1, n_steps, n_features))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            # Jika temp_input belum memiliki cukup elemen, reshape apa yang ada
            # Ini adalah kondisi fallback, idealnya temp_input selalu memiliki n_steps elemen
            padded_input = np.pad(temp_input, (0, n_steps - len(temp_input)), 'constant', constant_values=(0,))
            x_input = padded_input.reshape((1, n_steps, n_features))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            # Pastikan temp_input tidak tumbuh tak terkendali jika ada logika yang salah
            if len(temp_input) > n_steps * 2 : # Safety break
                 temp_input = temp_input[-n_steps:]
            lst_output.extend(yhat.tolist())
            i=i+1
            
    predictions_scaled = np.array(lst_output).reshape(-1,1)
    predictions = scaler.inverse_transform(predictions_scaled)
    
    # Dapatkan prediksi pada data latih untuk visualisasi
    train_predictions_scaled = model.predict(X, verbose=0)
    train_predictions = scaler.inverse_transform(train_predictions_scaled)

    return predictions.flatten(), train_predictions.flatten(), None

# --- UI Streamlit ---
st.set_page_config(layout="wide", page_title="Sistem Rekomendasi Tren Genre Film")

st.title("🎬 Sistem Rekomendasi Tren Genre Film")
st.markdown("""
Aplikasi ini menganalisis tren genre film berdasarkan data historis dan memprediksi tren masa depan menggunakan model LSTM.
Dataset yang digunakan [Film Genre Statistics on Kaggle](https://www.kaggle.com/datasets/thedevastator/film-genre-statistics/data)
""")

data = load_data('ThrowbackDataThursday Week 11 - Film Genre Stats.csv')

if data is not None and not data.empty:
    st.sidebar.success("Dataset berhasil dimuat!")

    # Pilihan Genre
    if 'Genre' not in data.columns:
        st.error("Kolom 'Genre' tidak ditemukan dalam dataset. Tidak dapat melanjutkan.")
    else:
        unique_genres = sorted(data['Genre'].unique())
        selected_genre = st.sidebar.selectbox("Pilih Genre", unique_genres)

        # Pilihan Metrik
        available_metrics = {
            "Pendapatan Kotor Disesuaikan Inflasi": "Inflation-Adjusted Gross",
            "Tiket Terjual": "Tickets Sold",
            "Jumlah Film Dirilis": "Movies Released",
            "Pendapatan Kotor (Nominal)": "Gross"
        }

        # Filter metrik yang ada di data
        valid_metrics_display = {k: v for k, v in available_metrics.items() if v in data.columns}

        if not valid_metrics_display:
            st.error("Tidak ada kolom metrik yang valid ('Inflation-Adjusted Gross', 'Tickets Sold', 'Movies Released', 'Gross') ditemukan dalam dataset.")
        else:
            selected_metric_display = st.sidebar.selectbox("Pilih Metrik untuk Analisis Tren", list(valid_metrics_display.keys()))
            metric_column = valid_metrics_display[selected_metric_display]

            # Filter data untuk genre dan metrik yang dipilih
            genre_data = data[data['Genre'] == selected_genre].sort_values('Year')

            if genre_data.empty or genre_data[metric_column].isnull().all():
                st.warning(f"Tidak ada data atau semua nilai kosong untuk metrik '{selected_metric_display}' pada genre '{selected_genre}'.")
            else:
                genre_data = genre_data[['Year', metric_column]].dropna(subset=[metric_column])
                genre_data = genre_data.drop_duplicates(subset=['Year'])  # Pastikan tahun unik

                st.header(f"Genre: {selected_genre}")
                st.header(f"Metrik: {selected_metric_display}")

                # Visualisasi Tren Historis
                fig_hist = px.line(genre_data, x='Year', y=metric_column, 
                                   title=f"Tren Historis {selected_metric_display} untuk {selected_genre}",
                                   labels={'Year': 'Tahun', metric_column: selected_metric_display})
                fig_hist.update_layout(xaxis_title='Tahun', yaxis_title=selected_metric_display)
                st.plotly_chart(fig_hist, use_container_width=True)

                # Prediksi LSTM
                if len(genre_data) >= 5:  # Minimal data untuk LSTM sederhana
                    with st.spinner(f"Melatih model LSTM dan membuat prediksi untuk {selected_genre}... Ini mungkin memakan waktu beberapa saat."):
                        future_preds, train_preds, error_msg = train_and_predict_lstm(
                            genre_data, metric_column, n_steps=3, epochs=100, future_predictions=3)

                    if error_msg:
                        st.warning(error_msg)
                    elif future_preds is not None and train_preds is not None:
                        st.subheader("Prediksi Tren Menggunakan LSTM")

                        last_hist_year = genre_data['Year'].max()

                        # Data historis
                        plot_df = genre_data[['Year', metric_column]].copy()
                        plot_df.rename(columns={metric_column: 'Nilai Aktual'}, inplace=True)
                        plot_df['Jenis'] = 'Historis'

                        # Data prediksi masa depan
                        future_years = np.arange(last_hist_year + 1, last_hist_year + 1 + len(future_preds))
                        future_df = pd.DataFrame({
                            'Year': future_years,
                            'Nilai': future_preds,
                            'Jenis': 'Prediksi Masa Depan (LSTM)'
                        })

                        # Gabungkan semua data untuk plot
                        combined_plot_df = pd.concat([
                            plot_df.rename(columns={'Nilai Aktual': 'Nilai'}),
                            future_df
                        ], ignore_index=True)

                        fig_pred = px.line(combined_plot_df, x='Year', y='Nilai', color='Jenis',
                                           title=f"Prediksi {selected_metric_display} untuk {selected_genre} (LSTM)",
                                           labels={'Year': 'Tahun', 'Nilai': selected_metric_display},
                                           markers=True)
                        fig_pred.update_layout(xaxis_title='Tahun', yaxis_title=selected_metric_display)
                        st.plotly_chart(fig_pred, use_container_width=True)

                        st.write("Prediksi nilai untuk beberapa tahun ke depan")
                        pred_table = pd.DataFrame({
                            'Tahun': future_years,
                            f'Prediksi {selected_metric_display}': future_preds
                        })
                        st.dataframe(pred_table.style.format({f'Prediksi {selected_metric_display}': "{:,.0f}"}))
                    else:
                        st.info("Tidak dapat membuat prediksi LSTM untuk genre ini dengan data saat ini.")
                else:
                    st.info(f"Tidak cukup data historis (minimal 5 tahun) untuk genre '{selected_genre}' untuk melakukan prediksi LSTM.")

                # Rekomendasi Film Teratas
                st.header(f"Film Teratas (Top Movies) untuk Genre ({selected_genre})")
                if 'Top Movie' in data.columns and 'Top Movie Inflation-Adjusted Gross (That Year)' in data.columns:
                    top_movies_genre = data[data['Genre'] == selected_genre].sort_values('Year', ascending=False)

                    top_movies_display = top_movies_genre[['Year', 'Top Movie', 'Top Movie Inflation-Adjusted Gross (That Year)']].copy()
                    top_movies_display.rename(columns={
                        'Year': 'Tahun Rilis',
                        'Top Movie': 'Judul Film Teratas',
                        'Top Movie Inflation-Adjusted Gross (That Year)': 'Pendapatan Film Teratas (Disesuaikan Inflasi)'
                    }, inplace=True)

                    top_movies_display = top_movies_display.drop_duplicates(subset=['Judul Film Teratas'], keep='first')

                    if not top_movies_display.empty:
                        st.dataframe(top_movies_display.head(15).style.format({'Pendapatan Film Teratas (Disesuaikan Inflasi)': "{:,.0f}"}), use_container_width=True)
                    else:
                        st.write("Tidak ada data film teratas untuk genre ini.")
                else:
                    st.warning("Kolom 'Top Movie' atau 'Top Movie Inflation-Adjusted Gross (That Year)' tidak ditemukan.")
else:
    st.info("Dataset tidak berhasil dimuat atau kosong.")

st.sidebar.markdown("---")
st.sidebar.markdown("Dibuat dengan Streamlit dan TensorFlow/Keras")