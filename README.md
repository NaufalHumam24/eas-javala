# eas-javala

Link Website : https://naufalhumam24.github.io/eas-javala/

---

# Javala — Pelestarian Cerita Rakyat Nusantara melalui Translasi dan TTS Bahasa Jawa

Javala adalah aplikasi web edukatif yang bertujuan melestarikan cerita rakyat Indonesia dengan menerjemahkannya ke dalam Bahasa Jawa dan mengubahnya menjadi suara (Text-to-Speech). Proyek ini menggabungkan *Frontend Web, **Backend NLP, dan **Teknologi TTS Bahasa Jawa*.

---

## 📌 Fitur Utama

- 🌍 Translasi otomatis cerita rakyat dari Bahasa Indonesia ke Bahasa Jawa menggunakan model NLP.
- 🔊 Pembacaan otomatis hasil cerita menggunakan TTS (Text-to-Speech) Bahasa Jawa.
- 🖼 Antarmuka web interaktif yang menampilkan cerita dan audio.
- 📖 Halaman khusus untuk tiap cerita rakyat.
- 📁 Sistem modular untuk menambahkan cerita baru secara mudah.

---

## 🖥 Teknologi yang Digunakan

### Frontend
- HTML5, CSS3, JavaScript (Vanilla)
- Responsive Web Design
- Gradio UI (opsional untuk integrasi demo AI)

### Backend (NLP & TTS)
- Python 
- Translasi Bahasa Indonesia → Bahasa Jawa menggunakan model fine-tuned mT5 Cendol.
- TTS Bahasa Jawa menggunakan model Facebook facebook/mms-tts-ind fine-tuned untuk Bahasa Jawa.
- Hugging Face Transformers, Torch

---

## 📁 Struktur Proyek

```

eas-javala/
├── frontend/
│   ├── index.html
│   ├── cerita.html
│   ├── tentang.html
│   ├── style.css
│   └── script.js
│
├── backend/
│   ├── app.py          
│   ├── backend.py
│   ├── best_model_mt5/
│   │   ├── adapter_config.json       
│   │   └── adapter_model.safetensors
│   │   └── special_tokens_map.json
│   │   └── spiece.model
│   │   └── tokenizer_config.json
│   │   └── training_args.bin
|   ├── cerita/
│   │   └── dataset_cerita_rakyat.csv
│   └── requirements.txt
│
└── README.md

```

---

## 📡 Endpoint API

| Endpoint     | Method | Fungsi                                     |
| ------------ | ------ | ------------------------------------------ |
| `/translate` | POST   | Menerjemahkan teks Indo → Jawa             |
| `/tts`       | POST   | Mengubah teks Bahasa Jawa jadi audio (WAV) |
| `/cerita`    | GET    | Mengambil daftar cerita (jika di-database) |

Contoh payload:

```json
POST /translate
{
  "text": "Pada zaman dahulu ada seekor harimau..."
}
```

---

## 📚 Contoh Studi Kasus

* Cerita rakyat Timun Mas diterjemahkan otomatis ke Bahasa Jawa dan disuarakan sebagai audio agar anak-anak atau peneliti budaya bisa mendengarkan versi lokal.
* Edukasi interaktif untuk sekolah dasar mengenai kearifan lokal dengan bantuan AI.

---

## 🔮 Pengembangan Selanjutnya

* Penambahan pilihan dialek Bahasa Jawa (Ngoko, Krama, dll).
* Deteksi otomatis Bahasa Indonesia pada input.
* Fitur bookmark & favoriting cerita favorit.
* Deploy ke Hugging Face Spaces untuk demo publik.

---
