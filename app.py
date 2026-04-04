from flask import Flask, Response, jsonify, render_template, request, make_response
import cv2
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
import numpy as np
import pickle
from tensorflow import keras
from collections import deque
import time
from datetime import datetime
import threading
import atexit
import subprocess
import queue
import os
import sys
import platform

# ── TTS setup (cross-platform) ─────────────────────────────────────────────────
tts_queue = queue.Queue()

def tts_worker():
    """Single persistent TTS thread — works on macOS and Linux (Render)"""
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            if platform.system() == 'Darwin':
                # macOS — native say command
                subprocess.run(['say', '-r', '150', text],
                               capture_output=True, timeout=10)
            else:
                # Linux (Render) — espeak
                subprocess.run(['espeak', '-s', '150', '-v', 'en', text],
                               capture_output=True, timeout=10)
        except Exception as e:
            print(f"[TTS] Error: {e}")

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak(text):
    """Queue text for speech — clears old pending items first"""
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
        except:
            pass
    tts_queue.put(text)


app = Flask(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
# Works both locally and on Render — models sit in ./Model/ relative to app.py
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR         = os.path.join(BASE_DIR, 'Model')

WORD_MODEL_PATH   = os.path.join(MODEL_DIR, 'word_model.h5')
WORD_LABEL_PATH   = os.path.join(MODEL_DIR, 'word_label_mapping.pkl')
LETTER_MODEL_PATH = os.path.join(MODEL_DIR, 'letter_model.h5')
LETTER_LABEL_PATH = os.path.join(MODEL_DIR, 'letter_label_mapping.pkl')

# ── Configuration ──────────────────────────────────────────────────────────────
WORD_BUFFER_SIZE = 60

# ── OpenCV overlay colours (BGR) ───────────────────────────────────────────────
COLOR_BG              = (40,  40,  40)
COLOR_TEXT            = (255, 255, 255)
COLOR_WORD            = (0,   255, 0)
COLOR_LETTER          = (100, 200, 255)
COLOR_CONFIDENCE_HIGH = (0,   255, 0)
COLOR_CONFIDENCE_MED  = (0,   165, 255)
COLOR_CONFIDENCE_LOW  = (0,   0,   255)


# ── Translator class ───────────────────────────────────────────────────────────
class WebHybridTranslator:

    def __init__(self):
        print("\n[LOADING MODELS]")

        try:
            self.word_model = keras.models.load_model(WORD_MODEL_PATH)
            with open(WORD_LABEL_PATH, 'rb') as f:
                self.word_mapping = pickle.load(f)
            print(f"  ✓ Word model   : {list(self.word_mapping.values())}")
        except Exception as e:
            print(f"  ✗ Word model failed: {e}")
            self.word_model   = None
            self.word_mapping = {}

        try:
            self.letter_model = keras.models.load_model(LETTER_MODEL_PATH)
            with open(LETTER_LABEL_PATH, 'rb') as f:
                self.letter_mapping = pickle.load(f)
            print(f"  ✓ Letter model : {list(self.letter_mapping.values())}")
        except Exception as e:
            print(f"  ✗ Letter model failed: {e}")
            self.letter_model   = None
            self.letter_mapping = {}

        if self.word_model is None and self.letter_model is None:
            raise Exception("No models loaded! Check Model/ directory.")

        # MediaPipe
        self.mp_hands = mp_hands
        self.hands    = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw           = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles

        # Mode
        self.mode = "LETTER" if self.letter_model else "WORD"

        # Detection state
        self.sequence_buffer    = deque(maxlen=WORD_BUFFER_SIZE)
        self.current_detection  = ""
        self.current_confidence = 0.0
        self.detection_type     = ""

        # Sentence builder
        self.sentence            = []
        self.current_word        = []
        self.last_letter_time    = 0
        self.letter_word_timeout = 2.0
        self.last_word_added     = ""
        self.last_word_add_time  = 0
        self.word_add_cooldown   = 3.0

        # Session log
        self.session_log   = []
        self.session_start = datetime.now()

        # FPS
        self.fps             = 0
        self.frame_count     = 0
        self.last_fps_update = time.time()

        # Letter hold logic
        self.letter_hold_start    = 0
        self.letter_hold_duration = 1.2
        self.last_letter          = ""

        # Camera / threading
        self.camera     = None
        self.is_running = False
        self.lock       = threading.Lock()

        # Word prediction config
        self.last_word_time          = 0
        self.word_check_interval     = 0.15
        self.confidence_threshold    = 0.80
        self.last_predicted_word     = ""
        self.word_confidence_history = deque(maxlen=5)
        self.stable_word_threshold   = 1

        # TTS cooldown
        self.last_spoken      = ""
        self.last_spoken_time = 0
        self.speak_cooldown   = 2.0

    # ── Cleanup ────────────────────────────────────────────────────────────────
    def cleanup(self):
        with self.lock:
            self.is_running = False
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            if self.hands is not None:
                self.hands.close()
                self.hands = None
        print("✓ Resources released")

    # ── Landmark extraction ────────────────────────────────────────────────────
    def normalize_landmarks(self, features):
        features  = features.reshape(21, 3)
        wrist     = features[0].copy()
        features  = features - wrist
        hand_size = np.linalg.norm(features[12] - features[0])
        if hand_size > 1e-6:
            features = features / hand_size
        return features.flatten()

    def extract_landmarks(self, frame):
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            features  = []
            for lm in landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
            features = self.normalize_landmarks(np.array(features))
            return features, landmarks
        return None, None

    # ── Prediction ─────────────────────────────────────────────────────────────
    def predict_letter(self, features):
        if self.letter_model is None:
            return None, 0.0
        inp   = np.expand_dims(features, axis=0)
        preds = self.letter_model.predict(inp, verbose=0)[0]
        idx   = np.argmax(preds)
        return self.letter_mapping[idx], float(preds[idx])

    def predict_word(self):
        if self.word_model is None or len(self.sequence_buffer) < WORD_BUFFER_SIZE:
            return None, 0.0
        sequence = np.expand_dims(np.array(list(self.sequence_buffer)), axis=0)
        preds    = self.word_model.predict(sequence, verbose=0)[0]
        idx      = np.argmax(preds)
        word     = self.word_mapping[idx]
        conf     = float(preds[idx])

        self.word_confidence_history.append((word, conf))
        if len(self.word_confidence_history) >= 2:
            counts = {}
            for w, c in self.word_confidence_history:
                if c >= self.confidence_threshold:
                    counts[w] = counts.get(w, 0) + 1
            if counts.get(word, 0) >= self.stable_word_threshold:
                return word, conf
        return None, 0.0

    # ── TTS helper ─────────────────────────────────────────────────────────────
    def try_speak(self, text, now):
        if (text != self.last_spoken or
                now - self.last_spoken_time > self.speak_cooldown):
            speak(text)
            self.last_spoken      = text
            self.last_spoken_time = now

    # ── Session log ────────────────────────────────────────────────────────────
    def log_detection(self, text, dtype, confidence=None):
        self.session_log.append({
            'time':       datetime.now().strftime('%H:%M:%S'),
            'text':       text.upper(),
            'type':       dtype,
            'confidence': round(confidence * 100) if confidence else None,
        })

    def build_log_file(self):
        lines = [
            "=" * 50,
            "  SIGN LANGUAGE TRANSLATION SESSION LOG",
            "=" * 50,
            f"  Date : {self.session_start.strftime('%Y-%m-%d')}",
            f"  Start: {self.session_start.strftime('%H:%M:%S')}",
            f"  End  : {datetime.now().strftime('%H:%M:%S')}",
            "=" * 50, "",
        ]
        if not self.session_log:
            lines.append("  No detections recorded.")
        else:
            lines.append(f"  {'TIME':<10} {'TYPE':<8} {'CONF':<6} TEXT")
            lines.append("  " + "-" * 40)
            for e in self.session_log:
                conf_str = f"{e['confidence']}%" if e['confidence'] else "  —  "
                lines.append(
                    f"  [{e['time']}]  {e['type']:<8} {conf_str:<6}  {e['text']}"
                )
        lines += [
            "", "=" * 50, "  FULL SENTENCE", "=" * 50,
            f"  {' '.join(self.sentence) if self.sentence else '(empty)'}",
            "", "=" * 50,
            f"  Total detections: {len(self.session_log)}",
            "=" * 50,
        ]
        return '\n'.join(lines)

    # ── Sentence builder ───────────────────────────────────────────────────────
    def add_letter_to_word(self, letter, now):
        if not self.current_word or self.current_word[-1] != letter:
            self.current_word.append(letter)
        self.last_letter_time = now

    def check_word_timeout(self, now):
        if (self.current_word and
                self.last_letter_time > 0 and
                now - self.last_letter_time > self.letter_word_timeout):
            word = ''.join(self.current_word)
            self.sentence.append(word)
            self.log_detection(word, 'LETTER')
            self.current_word     = []
            self.last_letter_time = 0
            speak(word)

    def add_word_to_sentence(self, word, conf, now):
        if (word != self.last_word_added or
                now - self.last_word_add_time > self.word_add_cooldown):
            self.sentence.append(word.upper())
            self.log_detection(word, 'WORD', conf)
            self.last_word_added    = word
            self.last_word_add_time = now

    def get_sentence_display(self):
        parts = self.sentence.copy()
        if self.current_word:
            parts.append(''.join(self.current_word) + '...')
        return ' '.join(parts)

    # ── Drawing ────────────────────────────────────────────────────────────────
    def get_hand_center(self, hand_landmarks, frame_shape):
        h, w = frame_shape[:2]
        cx   = int(np.mean([lm.x * w for lm in hand_landmarks.landmark]))
        cy   = int(np.mean([lm.y * h for lm in hand_landmarks.landmark])) - 80
        return cx, cy

    def draw_floating_text(self, frame, text, position, confidence, dtype):
        x, y       = position
        text_color = COLOR_LETTER if dtype == "LETTER" else COLOR_WORD
        conf_color = (COLOR_CONFIDENCE_HIGH if confidence > 0.7
                      else COLOR_CONFIDENCE_MED if confidence > 0.5
                      else COLOR_CONFIDENCE_LOW)
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2.0 if dtype == "LETTER" else 1.5
        thick = 4
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        pad = 20
        x1, y1 = x - tw//2 - pad, y - th - pad
        x2, y2 = x + tw//2 + pad, y + pad
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), text_color, 3)
        cv2.putText(frame, text, (x - tw//2, y), font, scale, text_color, thick)
        conf_str = f"{confidence*100:.0f}%"
        (cw, ch), _ = cv2.getTextSize(conf_str, font, 0.6, 2)
        cv2.putText(frame, conf_str, (x - cw//2, y2 + ch + 10), font, 0.6, conf_color, 2)
        type_str = f"[{dtype}]"
        (tw2, _), _ = cv2.getTextSize(type_str, font, 0.5, 2)
        cv2.putText(frame, type_str, (x - tw2//2, y1 - 10), font, 0.5, (200, 200, 200), 2)

    def draw_mode_indicator(self, frame):
        h, w       = frame.shape[:2]
        mode_color = COLOR_LETTER if self.mode == "LETTER" else COLOR_WORD
        cv2.putText(frame, f"Mode: {self.mode}", (w - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        cv2.putText(frame, f"FPS: {self.fps:.0f}", (w - 250, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
        if self.mode == "WORD":
            pct  = (len(self.sequence_buffer) / WORD_BUFFER_SIZE) * 100
            bcol = COLOR_WORD if len(self.sequence_buffer) >= WORD_BUFFER_SIZE else (0, 165, 255)
            cv2.putText(frame, f"Buffer: {pct:.0f}%", (w - 250, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bcol, 2)

    # ── Frame processing ───────────────────────────────────────────────────────
    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_update >= 1.0:
            self.fps             = self.frame_count
            self.frame_count     = 0
            self.last_fps_update = now

        self.check_word_timeout(now)
        features, hand_landmarks = self.extract_landmarks(frame)

        if features is not None:
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

            if self.mode == "LETTER":
                letter, conf = self.predict_letter(features)
                if letter and conf > 0.5:
                    if letter == self.last_letter:
                        if now - self.letter_hold_start >= self.letter_hold_duration:
                            self.current_detection  = letter
                            self.current_confidence = conf
                            self.detection_type     = "LETTER"
                            self.try_speak(letter, now)
                            self.add_letter_to_word(letter, now)
                    else:
                        self.last_letter        = letter
                        self.letter_hold_start  = now
                        self.current_detection  = letter
                        self.current_confidence = conf * 0.7
                        self.detection_type     = "LETTER"

            elif self.mode == "WORD":
                self.sequence_buffer.append(features)
                if (len(self.sequence_buffer) >= WORD_BUFFER_SIZE
                        and now - self.last_word_time >= self.word_check_interval):
                    word, conf = self.predict_word()
                    self.last_word_time = now
                    if word and conf >= self.confidence_threshold:
                        self.current_detection   = word
                        self.current_confidence  = conf
                        self.detection_type      = "WORD"
                        self.last_predicted_word = word
                        self.try_speak(word, now)
                        self.add_word_to_sentence(word, conf, now)

            if self.current_detection:
                center = self.get_hand_center(hand_landmarks, frame.shape)
                self.draw_floating_text(frame, self.current_detection,
                                        center, self.current_confidence,
                                        self.detection_type)
        else:
            self.letter_hold_start = 0
            self.last_letter       = ""
            if self.mode == "WORD":
                self.word_confidence_history.clear()

        self.draw_mode_indicator(frame)
        return frame


# ── Global state ───────────────────────────────────────────────────────────────
translator  = None
camera_lock = threading.Lock()

def cleanup_resources():
    global translator
    if translator is not None:
        translator.cleanup()

atexit.register(cleanup_resources)

def initialize_translator():
    global translator
    if translator is None:
        try:
            translator = WebHybridTranslator()
            print("✓ Translator initialized")
        except Exception as e:
            print(f"✗ Init failed: {e}")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    initialize_translator()
    return render_template('index.html')

def generate_frames():
    global translator
    initialize_translator()
    with camera_lock:
        if translator.camera is None:
            translator.camera = cv2.VideoCapture(0)
            translator.camera.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            translator.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            translator.is_running = True
    try:
        while translator.is_running:
            with camera_lock:
                if translator.camera is None:
                    break
                success, frame = translator.camera.read()
            if not success:
                break
            frame = translator.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        with camera_lock:
            if translator.camera is not None:
                translator.camera.release()
                translator.camera = None
        translator.is_running = False

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_state')
def get_state():
    global translator
    if translator:
        return jsonify({
            'mode':                translator.mode,
            'detection':           translator.current_detection,
            'confidence':          float(translator.current_confidence),
            'detection_type':      translator.detection_type,
            'sentence':            translator.get_sentence_display(),
            'sentence_words':      translator.sentence,
            'current_word':        ''.join(translator.current_word),
            'fps':                 translator.fps,
            'word_model_loaded':   translator.word_model  is not None,
            'letter_model_loaded': translator.letter_model is not None,
            'buffer_fill':         len(translator.sequence_buffer),
            'last_spoken':         translator.last_spoken,
            'log_count':           len(translator.session_log),
            'recent_log':          translator.session_log[-5:],
        })
    return jsonify({'error': 'Translator not initialized'})

@app.route('/get_log')
def get_log():
    global translator
    if translator:
        return jsonify({
            'log':           translator.session_log,
            'session_start': translator.session_start.strftime('%Y-%m-%d %H:%M:%S'),
            'total':         len(translator.session_log),
        })
    return jsonify({'error': 'Translator not initialized'})

@app.route('/download_log')
def download_log():
    global translator
    if translator:
        content  = translator.build_log_file()
        filename = f"session_{translator.session_start.strftime('%Y%m%d_%H%M%S')}.txt"
        response = make_response(content)
        response.headers['Content-Type']        = 'text/plain'
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        return response
    return jsonify({'error': 'not initialized'})

@app.route('/toggle_mode', methods=['POST'])
def toggle_mode():
    global translator
    if translator:
        if translator.mode == "LETTER" and translator.word_model:
            translator.mode = "WORD"
        elif translator.mode == "WORD" and translator.letter_model:
            translator.mode = "LETTER"
        else:
            missing = "word" if translator.mode == "LETTER" else "letter"
            return jsonify({'success': False, 'reason': f'{missing} model not loaded'})
        translator.sequence_buffer.clear()
        translator.current_detection  = ""
        translator.current_confidence = 0.0
        translator.word_confidence_history.clear()
        return jsonify({'success': True, 'mode': translator.mode})
    return jsonify({'success': False})

@app.route('/speak_sentence', methods=['POST'])
def speak_sentence():
    global translator
    if translator:
        full = translator.get_sentence_display().replace('...', '').strip()
        if full:
            speak(full)
            return jsonify({'success': True, 'spoken': full})
    return jsonify({'success': False})

@app.route('/backspace', methods=['POST'])
def backspace():
    global translator
    if translator:
        if translator.current_word:
            translator.current_word.pop()
        elif translator.sentence:
            translator.sentence.pop()
        return jsonify({'success': True,
                        'sentence': translator.get_sentence_display()})
    return jsonify({'success': False})

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global translator
    if translator:
        translator.sentence.clear()
        translator.current_word       = []
        translator.current_detection  = ""
        translator.current_confidence = 0.0
        translator.sequence_buffer.clear()
        translator.word_confidence_history.clear()
        translator.last_spoken        = ""
        translator.last_letter_time   = 0
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/cleanup', methods=['POST'])
def cleanup():
    global translator
    if translator:
        translator.cleanup()
    return jsonify({'success': True})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5008))
    print("\n" + "=" * 60)
    print("  SIGN LANGUAGE TRANSLATOR")
    print(f"  Platform : {platform.system()}")
    print(f"  Port     : {port}")
    print(f"  Open     : http://localhost:{port}")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
