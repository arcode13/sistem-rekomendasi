# Sistem Pemberi Rekomendasi Tren Genre Film dengan Deep Learning (LSTM)

Proyek ini bertujuan untuk menganalisis tren historis genre film dan memprediksi tren masa depan menggunakan model *deep learning* Long Short-Term Memory (LSTM). Aplikasi ini dibangun dengan Python dan Streamlit, menyediakan antarmuka pengguna grafis yang interaktif.

[🔗 Coba Aplikasi Live](http://103.160.213.26:8501)

## Dataset

* **Nama Dataset:** Film Genre Statistics
* **Sumber:** [Kaggle - Film Genre Statistics](https://www.kaggle.com/datasets/thedevastator/film-genre-statistics/data)
* **Deskripsi Singkat:** Dataset ini berisi statistik genre untuk film yang dirilis antara tahun 1995 dan 2018. Dataset ini menyediakan informasi tentang berbagai aspek film, seperti pendapatan kotor, tiket terjual, dan angka yang disesuaikan dengan inflasi. Kolom-kolomnya meliputi genre, tahun rilis, jumlah film yang dirilis, total pendapatan kotor, total tiket terjual, pendapatan kotor yang disesuaikan inflasi, judul film terlaris, pendapatan film terlaris, dan pendapatan film terlaris yang disesuaikan inflasi.

## Metodologi dan Implementasi

Sistem ini dikembangkan menggunakan Python dengan beberapa pustaka utama

* **Streamlit:** Untuk membangun antarmuka pengguna grafis (GUI) berbasis web.
* **Pandas & NumPy:** Untuk manipulasi, pembersihan, dan pra-pemrosesan data.
* **Plotly Express:** Untuk visualisasi data interaktif (grafik tren).
* **Scikit-learn:** Untuk normalisasi data (menggunakan `MinMaxScaler`).
* **TensorFlow (Keras API):** Untuk membangun, melatih, dan melakukan prediksi dengan model *deep learning* LSTM.

**Alur Kerja Aplikasi**

1.  **Pemuatan dan Pra-pemrosesan Data**
    * Load dataset CSV secara otomatis.
    * Data dibersihkan: kolom numerik dikonversi, karakter non-numerik dihapus, dan nilai yang hilang ditangani.
2.  **Pemilihan Pengguna**
    * Pengguna memilih genre film dan metrik kinerja (misalnya, "Pendapatan Kotor Disesuaikan Inflasi", "Tiket Terjual") yang ingin dianalisis.
3.  **Analisis Tren Historis**
    * Grafik garis interaktif menampilkan tren historis dari metrik yang dipilih untuk genre tersebut.
4.  **Pemodelan Prediktif dengan LSTM**
    * **Model:** Jaringan Long Short-Term Memory (LSTM) dipilih karena kemampuannya dalam menangani data sekuensial dan deret waktu, serta mempelajari dependensi jangka panjang.
    * **Persiapan Data:** Data metrik yang dipilih dinormalisasi ke rentang [0, 1]. Kemudian, data diubah menjadi format sekuens input-output untuk pelatihan LSTM.
    * **Arsitektur Model:** Model LSTM sederhana dengan satu lapisan LSTM diikuti oleh satu lapisan Dense untuk output.
    * **Pelatihan:** Model dilatih menggunakan data historis dari genre yang dipilih.
    * **Prediksi:** Model menghasilkan prediksi untuk beberapa tahun ke depan.
    * **Visualisasi Prediksi:** Prediksi ditampilkan pada grafik yang sama dengan data historis, serta dalam bentuk tabel.
5.  **Rekomendasi Film Teratas**
    * Menampilkan daftar film teratas (`Top Movie`) dari genre yang dipilih berdasarkan data historis.

## Fitur Aplikasi

* Load dataset CSV secara otomatis.
* Pemilihan genre film dan metrik analisis.
* Visualisasi tren historis metrik yang dipilih.
* Prediksi tren masa depan menggunakan model LSTM.
* Visualisasi gabungan data historis dan prediksi.
* Tampilan tabel prediksi nilai untuk tahun-tahun mendatang.
* Daftar film teratas untuk genre yang dipilih.

## Cara Menjalankan Aplikasi

1.  **Prasyarat**
    * Python 3.7 atau lebih tinggi.
    * `pip` (Python package installer).

2.  **Klon Repositori**
    ```bash
    git clone https://github.com/arcode13/sistem-rekomendasi
    cd sistem-rekomendasi
    ```

3.  **Instal Dependensi**
    Disarankan untuk membuat *virtual environment*.
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Linux/macOS
    # venv\Scripts\activate    # Untuk Windows
    ```
    Kemudian instal dependensi
    ```bash
    pip install -r requirements.txt
    ```

4.  **Siapkan Dataset**
    * Unduh dataset `ThrowbackDataThursday Week 11 - Film Genre Stats.csv` dari [Kaggle](https://www.kaggle.com/datasets/thedevastator/film-genre-statistics/data).
    * Pastikan file dataset berada di direktori yang sama dengan file `app.py`.

5.  **Jalankan Aplikasi Streamlit**
    Jika file utama Anda bernama `app.py`
    ```bash
    streamlit run app.py
    ```
    Aplikasi akan terbuka secara otomatis di browser web Anda.

## Struktur File

```
.
├── app.py                                  # Kode utama aplikasi Streamlit
├── "ThrowbackDataThursday Week 11 - Film Genre Stats.csv"  # File dataset
└── requirements.txt                        # Daftar dependensi Python
```