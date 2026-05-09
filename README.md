# Sign Language Translation System
 
A real-time, offline American Sign Language (ASL) recognition system that bridges communication between the deaf and hard-of-hearing community and the general hearing population. Built using a dual LSTM architecture, MediaPipe hand tracking, and a Flask web interface with Text-to-Speech output and session logging.
 
**Kantipur Engineering College — CT 755 Major Project**
Bachelor of Computer Engineering, Tribhuvan University — March 2026
 

 
## Features
 
- Real-time ASL letter recognition (A–Z) using single-frame LSTM
- Real-time ASL word recognition (8 words) using Bidirectional LSTM on 60-frame sequences
- Automatic mode switching between Letter and Word mode based on hand motion analysis
- Text-to-Speech output for every confirmed detection (macOS + Linux)
- Automatic sentence builder — letters accumulate into words, words accumulate into sentences
- Session history log with timestamps and confidence scores
- Downloadable session export as `.txt` file
- Fully offline — all inference runs on local hardware with no cloud dependency
 
---
 
## Supported ASL Words
 
| Word | Word | Word | Word |
|------|------|------|------|
| HELLO | CLEAN | BOOK | BEAUTIFUL |
| CAT | EAT | FATHER | LATER |
 
---
 ## Demo
 -linke (https://www.veed.io/view/62f0c6f6-50ee-4eb1-a5d4-a0e0a34e67f1?source=Dashboard&panel=share)

 ___
## Model Performance
 
| Model | Architecture | Input Shape | Training Accuracy | Test Accuracy | F1-Score | Inference Time |
|-------|-------------|-------------|:-----------------:|:-------------:|:--------:|:--------------:|
| Letter Recognition | 2-layer LSTM | (1, 1, 63) | 97.8% | 95.9% | 0.951 | 1 ms/frame |
| Word Recognition | Conv1D + BiLSTM | (1, 60, 63) | 84.7% | 85.1% | 0.826 | 1 ms/frame |
 
---
 
## Model Architecture
 
### Letter Recognition Model (Static Gestures A–Z)
 
The letter model uses a two-layer LSTM network designed to classify static hand configurations. Although letter gestures are inherently single-frame, the LSTM architecture was retained to maintain a consistent modeling framework across both recognition pipelines.
 
```
Input: (1, 1, 63) — single frame, 63 landmark features
    │
    ├── LSTM Layer 1 — 128 units, return_sequences=True
    │       Captures initial spatial patterns from normalized hand landmarks
    │
    ├── LSTM Layer 2 — 64 units
    │       Refines features into compact summary representation
    │
    ├── Dense — 128 units, ReLU activation
    │       Non-linear transformation for classification
    │
    └── Output — 26 units, Softmax
            Probability distribution over A–Z
```
 
**Training details:**
- Dataset: Kaggle ASL Alphabet dataset (26 classes, clean annotations)
- Epochs: 100
- Regularization: Dropout (rate=0.3) + Exposure augmentation during training
- Optimizer: Adam
- The gap between training (97.8%) and validation (96.4%) accuracy is intentional — dropout and exposure augmentation make training harder than validation, which is expected behavior under heavy regularization
 
**Key confusion pairs identified:** U↔V (most common), M↔N, Z↔W
 
---
 
### Word Recognition Model (Dynamic Gestures — 8 ASL Words)
 
The word model uses lightweight Conv1D layers for temporal preprocessing followed by a Bidirectional LSTM as the core classifier.
 
```
Input: (1, 60, 63) — 60-frame sequence at 30 FPS (~2 seconds)
    │
    ├── Conv1D — 24 filters, kernel=5, along time axis
    │       Captures short-range motion patterns across frames
    │       + BatchNormalization + MaxPooling + Dropout
    │
    ├── Conv1D — 48 filters, kernel=3
    │       Extracts higher-level temporal features
    │       + BatchNormalization + MaxPooling + Dropout
    │
    ├── Bidirectional LSTM — 32 units
    │       Forward pass: learns how gesture evolves over time
    │       Backward pass: incorporates later frames to resolve early ambiguity
    │       + Dropout + Recurrent Dropout
    │
    ├── Dense — 32 units, ReLU + L2 regularization
    │
    └── Output — 8 units, Softmax
            Probability distribution over 8 word classes
```
 
**Training details:**
- Dataset: 400 video sequences total, 50 sequences per word class
- Each sequence: 60 frames captured at 30 FPS
- Training sequences: 30-frame videos temporally interpolated to 60 frames
- Regularization: High dropout, batch normalization, data augmentation, class balancing
- Optimizer: Adam
- The Conv1D layers serve as a temporal preprocessing stage, not a standalone classifier
 
**Key confusion pairs identified:** Beautiful↔Eat, Beautiful↔Father (similar face-level motion trajectories)
 
---
 
### Architecture Comparison
 
| Metric | Letter LSTM | CNN (baseline) |
|--------|:-----------:|:--------------:|
| Test Accuracy | **95.9%** | 86.0% |
| Validation Accuracy | **96.4%** | 84.0% |
| Training Accuracy | **97.0%** | 88.0% |
 
| Metric | Word BiLSTM | CNN+LSTM (baseline) |
|--------|:-----------:|:-------------------:|
| Test Accuracy | 88.5% | **93.0%** |
| Validation Accuracy | **89.0%** | 92.2% |
| Training Accuracy | 88.0% | 92.4% |
 
The CNN+LSTM achieves higher raw test accuracy but overfits on the small 50-sequence-per-class dataset. The BiLSTM was chosen for its better generalization, lower computational cost, and architectural consistency with the letter model.
 
---
 
## Preprocessing Pipeline
 
Every frame goes through the following stages before reaching the model:
 
```
Webcam Frame
    │
    ├── 1. BGR → RGB conversion (OpenCV default → MediaPipe requirement)
    │
    ├── 2. Frame resize (standardize spatial resolution)
    │
    ├── 3. Exposure augmentation (training only — multiple brightness levels)
    │
    ├── 4. MediaPipe BlazePalm — palm detection via SSD architecture
    │
    ├── 5. MediaPipe Hand Landmark Model — 21 3D keypoints extracted
    │
    ├── 6. Landmark normalization
    │       - Wrist set as origin
    │       - All coordinates relative to wrist
    │       - Scaled by wrist-to-middle-fingertip distance
    │       - Result: scale, position, and distance invariant
    │
    └── 7. Feature vector — 21 landmarks × 3 coordinates = 63 features
```
 
---
 
## System Pipeline
 
```
Webcam → OpenCV → MediaPipe → 63-dim normalized landmark vector
                                        │
                    ┌───────────────────┴──────────────────────┐
               STILL hand                                  MOVING hand
               Letter Mode                                 Word Mode
               Input: (1,1,63)                         Input: (1,60,63)
               2-layer LSTM                            Conv1D + BiLSTM
               Predicts A–Z                            Predicts word
                    └───────────────────┬──────────────────────┘
                                        │
                          Temporal smoothing + confidence check (≥80%)
                                        │
                          ┌─────────────┼─────────────┐
                        TTS           Sentence      Session
                       speak          builder        log
                                        │
                                 Flask → Browser UI
```
 
---
 
## Mode Switching Logic
 
The system automatically switches between Letter and Word mode by monitoring hand displacement across consecutive frames.
 
- **Enter Word Mode** — if 6 out of last 10 frames exceed movement threshold (0.015 normalized units)
- **Enter Letter Mode** — if 8 out of last 10 frames show no significant movement
- **Cooldown** — minimum 1.0 second between mode switches to prevent rapid oscillation
- **Asymmetric design** — more consecutive still frames required to exit Word mode, preventing premature interruption of a word gesture
 
---
 
## Prediction Stabilization
 
**Letter Mode:**
- Confidence threshold: 50% minimum to consider a prediction valid
- Hold timer: letter must be detected consistently for 1.2 seconds before confirmed
- Tentative display at 70% confidence while hold timer is running
- Resets on hand loss
 
**Word Mode:**
- Sliding buffer of 60 frames (deque)
- Prediction attempted every 0.15 seconds when buffer is full
- Confidence history of last 5 predictions — word accepted only when it appears with ≥80% confidence
- Confidence history clears on hand loss
 
---
 
## Tech Stack
 
| Layer | Technology |
|-------|-----------|
| Hand Detection | MediaPipe Hands (BlazePalm + Hand Landmark Model) |
| Feature Extraction | 21 3D landmarks → 63-dim normalized vector |
| Letter Model | 2-layer LSTM |
| Word Model | Conv1D + Bidirectional LSTM |
| Web Framework | Flask |
| Video Streaming | MJPEG over HTTP multipart |
| Text-to-Speech | macOS `say` / Linux `espeak` |
| Threading | Python `threading` + `queue` for async inference |
| Frontend | HTML / CSS / Vanilla JavaScript |
| Model Format | Keras `.h5` |
| Version Control | Git + Git LFS (for model files) |
 
---
 
## Project Structure
 
```
sign-language-translation-system/
├── app.py                        # Flask backend — translator logic + all API routes
├── templates/
│   └── index.html                # Frontend UI
├── model/
│   ├── letter_model.h5           # Trained letter LSTM       [Git LFS]
│   ├── letter_label_mapping.pkl  # Letter class mapping      [Git LFS]
│   ├── word_model.h5             # Trained word BiLSTM       [Git LFS]
│   └── word_label_mapping.pkl    # Word class mapping        [Git LFS]
├── requirements.txt
├── .gitattributes                # Git LFS config for .h5 and .pkl
├── .gitignore
└── README.md
```
 
---
 
## Local Setup
 
### Requirements
- Python 3.10
- Webcam (720p or higher)
- macOS or Linux
 
### Installation
 
```bash
# Clone the repository
git clone https://github.com/yogeshY0/Sign-language-translation-System.git
cd Sign-language-translation-System
 
# Pull model files via Git LFS
git lfs install
git lfs pull
 
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
 
# Install dependencies
pip install -r requirements.txt
 
# Run
python app.py
```
 
Open `http://localhost:5008` in your browser.
 
---
 
## Hardware Requirements
 
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| CPU | Intel i3 / Apple M1 | Intel i5 / Apple M2 |
| Webcam | 720p | 1080p |
| Storage | 1 GB | 2 GB |
| GPU | Not required | Not required |
 
---
 
## API Endpoints
 
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main UI |
| `/video_feed` | GET | MJPEG video stream |
| `/get_state` | GET | Current detection, mode, confidence, log |
| `/toggle_mode` | POST | Switch between Letter and Word mode |
| `/speak_sentence` | POST | Speak full built sentence via TTS |
| `/backspace` | POST | Remove last letter or word from sentence |
| `/clear_sentence` | POST | Reset sentence and detection state |
| `/get_log` | GET | Full session log as JSON |
| `/download_log` | GET | Download session log as `.txt` file |
| `/cleanup` | POST | Release camera and MediaPipe resources |
 
---
 
## References
 
1. Harati, R. (2023). Importance of sign language in communication. *J. Commun. Disord.*, 11, 247.
2. Dhruv, A. J., & Bharti, S. K. (2021). Real-time sign language converter. *AIMV 2021*, 1–6.
3. Patel, K., & Sharma, S. (2022). Real-time offline sign language translation. *Int. J. Innov. Res. Eng.*, 1(2), 1–6.
4. Kumari, M., & Anand, V. (2023). Efficient real-time isolated sign language recognition using MobileNetV2 and attention LSTM. *JETIR*, 10(11), 332–336.
5. Kamble, S. (2025). SLRNet: A real-time LSTM-based sign language recognition system. *arXiv:2506.11154*.
 
---
 
*Submitted to the Department of Computer and Electronics Engineering, Kantipur Engineering College, Dhapakhel, Lalitpur — March 2026*
 
