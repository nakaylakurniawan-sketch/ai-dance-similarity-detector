# DanceAI — AI Dance Similarity Detector

Prototype website untuk membandingkan video dance referensi dengan video latihan menggunakan:
- MediaPipe Pose untuk mendeteksi landmark tubuh
- OpenCV untuk membaca video
- Normalisasi landmark
- Dynamic Time Warping (DTW) untuk membandingkan urutan gerakan
- Streamlit untuk antarmuka website

## Jalankan secara lokal

Disarankan Python 3.10 atau 3.11.

```bash
python -m venv .venv
```

Aktifkan environment:

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install:
```bash
pip install -r requirements.txt
```

Jalankan:
```bash
streamlit run app.py
```

Browser akan membuka alamat lokal Streamlit.

## Deploy online gratis dengan Streamlit Community Cloud

1. Buat akun GitHub jika belum punya.
2. Buat repository baru, misalnya `ai-dance-similarity-detector`.
3. Upload `app.py`, `requirements.txt`, dan `README.md` ke repository.
4. Buka https://share.streamlit.io/
5. Login menggunakan GitHub.
6. Pilih **Create app**, lalu pilih repository, branch, dan file utama `app.py`.
7. Klik **Deploy** dan tunggu proses instalasi selesai.

Setelah deploy berhasil, Streamlit akan memberikan URL aplikasi yang bisa dibuka melalui browser.

## Batasan prototype

- Skor adalah indikator heuristik dari jarak landmark dan DTW, bukan probabilitas akurasi atau penilaian profesional.
- Hasil dipengaruhi sudut kamera, pencahayaan, pakaian, tubuh yang tertutup, dan posisi kamera.
- Untuk demo, gunakan video pendek, satu orang, dan seluruh tubuh terlihat.
- Aplikasi mengambil sampel hingga 75 frame per video untuk membatasi beban komputasi.
- Maksimal ukuran upload yang disarankan adalah 50 MB per video.
- Video diproses saat analisis dan tidak disimpan sebagai database oleh kode aplikasi.
- Deployment gratis dapat memiliki batas resource dan waktu proses; jika aplikasi gagal start karena resource, gunakan video lebih pendek atau resolusi lebih kecil.

## Catatan metodologi

Skor kemiripan menggunakan transformasi heuristik:
`score = 100 * exp(-2 * normalized_DTW_distance)`

Skala ini dibuat agar hasil mudah dibaca, tetapi belum dikalibrasi atau divalidasi terhadap penilaian manusia. Jangan menyebutnya sebagai probabilitas atau akurasi model.
