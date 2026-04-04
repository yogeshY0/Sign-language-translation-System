# Sign Language Translation System

> Real-time American Sign Language (ASL) recognition system using dual LSTM architecture, MediaPipe hand tracking, and a Flask web interface — with Text-to-Speech output and session logging.

**Kantipur Engineering College — CT 755 Major Project**  
Bachelor of Computer Engineering, 2026



## Demo

> Open `http://localhost:5008` after running locally, or visit the deployed Render URL.

---

## Features

- **Real-time Letter Recognition** — Detects all 26 ASL alphabets (A–Z) using a single-frame LSTM model
- **Real-time Word Recognition** — Detects 8 ASL words using a Bidirectional LSTM on 60-frame sequences
- **Automatic Mode Switching** — Switches between Letter and Word mode based on hand motion analysis
- **Text-to-Speech** — Speaks every confirmed detection aloud (macOS + Linux/Render)
- **Sentence Builder** — Automatically accumulates letters into words and words into sentences
- **Session History** — Logs every detection with timestamp and confidence score
- **Export Log** — Download full session as a `.txt` file
- **Fully Offline** — All processing done locally, no cloud dependency
- **Cross-platform** — Runs on macOS and Linux (Render)

---

## Supported ASL Words

`HELLO` `CLEAN` `BOOK` `BEAUTIFUL` `CAT` `EAT` `FATHER` `LATER`

---

## Model Performance

| Model | Training Accuracy | Test Accuracy | F1-Score |
|-------|:-----------------:|:-------------:|:--------:|
| Letter LSTM (A–Z) | 97.8% | 95.9% | 0.951 |
| Word BiLSTM (8 words) | 84.7% | 85.1% | 0.826 |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Hand Detection | MediaPipe Hands (BlazePalm + Landmark Model) |
| Feature Extraction | 21 3D landmarks → 63-dim normalized vector |
| Letter Model | 2-layer LSTM (128 → 64 units) |
| Word Model | Conv1D + Bidirectional LSTM (32 units) |
| Backend | Flask (Python) |
| Frontend | HTML / CSS / JavaScript |
| TTS | macOS `say` / Linux `espeak` |
| Deployment | Render (gunicorn) |

---

## Project Structure

```
sign-language-translator/
├── app.py                  # Flask backend — all routes and translator logic
├── templates/
│   └── index.html          # Frontend UI
├── Model/
│   ├── letter_model.h5     # Trained letter LSTM (tracked via Git LFS)
│   ├── letter_label_mapping.pkl
│   ├── word_model.h5       # Trained word BiLSTM (tracked via Git LFS)
│   └── word_label_mapping.pkl
├── requirements.txt
├── render.yaml             # Render deployment config
├── .gitattributes          # Git LFS config for .h5 and .pkl files
├── .gitignore
└── README.md
```

---

## Local Setup

### Prerequisites
- Python 3.10
- Webcam
- macOS or Linux

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/sign-language-translator.git
cd sign-language-translator

# 2. Install Git LFS and pull model files
git lfs install
git lfs pull

# 3. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run
python app.py
```

Open your browser at `http://localhost:5008`

---

## Deployment on Render

### Steps

1. Push this repository to GitHub (with Git LFS enabled)
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repository
4. Render auto-detects `render.yaml` — no manual config needed
5. Click **Deploy**

> **Note:** Render's free tier sleeps after 15 minutes of inactivity. The first load after sleep takes ~30 seconds.

---

## How It Works

```
Webcam → OpenCV → MediaPipe → 63-dim landmark vector
                                      │
                    ┌─────────────────┴──────────────────┐
                    │ LETTER mode (still hand)            │ WORD mode (moving hand)
                    │ Single frame (1,1,63)               │ 60-frame sequence (1,60,63)
                    │ LSTM → A–Z                          │ BiLSTM → word
                    └─────────────────┬──────────────────┘
                                      │
                          Temporal smoothing + confidence check
                                      │
                          TTS speak + Sentence builder + Session log
                                      │
                               Flask → Browser UI
```

---

## Hardware Requirements

| | Minimum | Recommended |
|--|---------|-------------|
| RAM | 8 GB | 16 GB |
| CPU | Intel i3 | Intel i5 / Apple M1 |
| Webcam | 720p | 1080p |
| Storage | 1 GB | 2 GB |

---

## Limitations

- Word vocabulary limited to 8 words
- Single-hand gestures only
- Performance degrades in poor lighting
- No sentence-level NLP context

## Future Enhancements

- Expand word vocabulary using larger datasets
- Add NLP module for sentence correction
- Mobile app deployment
- Two-hand gesture support
- MediaPipe Holistic for facial expression recognition

---

## License

This project is submitted as an academic requirement for the Bachelor of Computer Engineering degree at Kantipur Engineering College, affiliated to Tribhuvan University.
