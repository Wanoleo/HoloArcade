# HoloEscape — AR Gesture-Controlled Escape Room

HoloEscape is a desktop webcam-based augmented-reality-style escape room built with Python, OpenCV, MediaPipe, scikit-learn, and Pygame. The player interacts with 2D virtual overlays rendered over a live camera feed using trained hand-gesture classifiers (KNN, SVM, and Random Forest).

> **Scope Note:** Version 1 is a 2D webcam-overlay prototype rendered in screen coordinates. It is not spatial AR, room-scale tracking, or headset AR.

---

## Project Structure

```text
HoloEscape/
├── dataset/
│   ├── raw/                  # Raw per-session gesture CSVs (with session_id & participant)
│   └── processed/            # Normalized train.csv and test.csv (grouped by session)
├── models/
│   ├── hand_landmarker.task  # MediaPipe HandLandmarker model
│   ├── knn_model.joblib
│   ├── svm_model.joblib
│   ├── random_forest_model.joblib
│   ├── best_model.joblib
│   └── model_metadata.json
├── results/
│   ├── metrics/              # model_comparison.csv, classification_reports.json, runtime_benchmark.json
│   └── plots/                # cm_*.png and model_comparison.png
├── assets/
├── src/
│   ├── config.py             # Shared gesture classes, feature schema, and paths
│   ├── camera.py             # Safe OpenCV webcam capture + Pygame surface conversion
│   ├── hand_detection.py     # MediaPipe HandLandmarker (21 landmarks -> 63 features)
│   ├── collect_data.py       # Interactive labelled gesture collector
│   ├── preprocessing.py      # Wrist-centered, scale-normalized preprocessing + session split
│   ├── train_models.py       # Trains & compares KNN, SVM, and Random Forest
│   ├── evaluate_models.py    # Generates classification metrics & confusion-matrix plots
│   ├── gesture_recognition.py# Live inference with confidence gating, smoothing & debounce
│   ├── game_logic.py         # ML-independent 3-stage escape-room state machine
│   ├── renderer.py           # 2D AR-style Pygame HUD and virtual puzzle overlays
│   └── benchmark.py          # Measures live FPS, hand-tracking time & ML latency
├── tests/                    # Unit & integration tests (M1–M6)
├── main.py                   # Game entry point
├── requirements.txt
└── README.md
```

---

## Setup (Windows PowerShell)

1. Create and activate the virtual environment:
   ```powershell
   python -m venv .venv --system-site-packages
   .\.venv\Scripts\Activate.ps1
   ```
   *(If PowerShell blocks script execution, you can run commands directly via `.\.venv\Scripts\python.exe`.)*

2. Install dependencies:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
   *(Note: `pygame-ce==2.5.8` is used as a drop-in replacement for `pygame` because it provides prebuilt Python 3.14 Windows wheels.)*

---

## Workflow Commands (Windows PowerShell)

### 1. Collect Gesture Data
Record across multiple sessions/people (target ~200–300 samples per class total):
```powershell
.\.venv\Scripts\python.exe -m src.collect_data --participant user1 --session s1
.\.venv\Scripts\python.exe -m src.collect_data --participant user1 --session s2
```
- Keys `1`–`6`: select gesture class (`open_palm`, `fist`, `pointing`, `thumbs_up`, `pinch`, `two_fingers`)
- `SPACE`: toggle continuous recording | `S`: capture single frame | `Q`/`ESC`: quit

### 2. Preprocess, Train & Compare Models (KNN, SVM, Random Forest)
```powershell
.\.venv\Scripts\python.exe -m src.train_models
.\.venv\Scripts\python.exe -m src.evaluate_models
```
This splits raw CSVs by `session_id` (preventing adjacent-frame leakage), trains all three classifiers, saves `models/best_model.joblib`, and writes metrics and confusion matrices to `results/`.

### 3. Play HoloEscape
```powershell
.\.venv\Scripts\python.exe main.py
```
Optional flags:
- `--model models\svm_model.joblib` — test any saved classifier
- `--camera 1` — select a different webcam index

**Keyboard shortcuts:** `R` to restart the level anytime, `Q` or `ESC` to quit.

### 4. Run Tests & Runtime Benchmark
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.benchmark --frames 60
```

---

## Gesture Controls & Level Guide

| Gesture | Action |
| :--- | :--- |
| `open_palm` | Open the on-screen controls menu |
| `fist` | Close menu / release artifact / clear exit keypad |
| `pointing` | Move pointer reticle & lock onto a virtual card, dial, or button |
| `thumbs_up` | Confirm / activate the selected item |
| `pinch` | Grab, hold, and drag the Holo-Cube (release to drop) |
| `two_fingers` | Rotate the Holo-Cube |

**Level Stages:**
1. **Combination Lock**: Point at `CLUE NOTE` + `thumbs_up` to reveal code `3 - 7 - 2`, set the 3 dials, then select `UNLOCK CHEST` + `thumbs_up`.
2. **Mystery Holo-Cube**: Pinch-grab the Holo-Cube, drag it into the `SCANNER` ring, and release to reveal the exit code `7 - 2 - 4`.
3. **Final Exit Door**: Enter `7 - 2 - 4` on the keypad (`pointing` + `thumbs_up`) to unlock the door and view your clear time.

---

## Measured Evaluation & Runtime Results

Evaluated on `2,045` held-out test samples (session-grouped split):

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| KNN (`n_neighbors=5`) | 0.6020 | 0.6443 | 0.6070 | 0.5903 |
| SVM (`RBF, C=1.0`) | 0.7159 | 0.7664 | 0.7096 | 0.7209 |
| **Random Forest (`100 trees, max_depth=15`)** | **0.8572** | **0.8809** | **0.8577** | **0.8576** |

Live runtime benchmark (`960x540`, 60 frames with `best_model.joblib`):
- **Measured FPS**: `30.8 FPS` (`32.46 ms` avg frame time, `13.31 ms` avg MediaPipe hand-tracking time)

---

## Lab 14 Concept Mapping

- **Lab 1 — Data Preprocessing**: Wrist-centered, hand-scale-normalized 63-feature preparation and session-grouped train/test splitting ([src/preprocessing.py](file:///d:/ML_Project/src/preprocessing.py)).
- **Lab 5 — Random Forest**: Trained and compared against KNN and SVM ([src/train_models.py](file:///d:/ML_Project/src/train_models.py)).
- **Lab 6 — SVM (RBF)**: Trained with `StandardScaler` pipeline ([src/train_models.py](file:///d:/ML_Project/src/train_models.py)).
- **Lab 8 — KNN**: Trained with `StandardScaler` pipeline ([src/train_models.py](file:///d:/ML_Project/src/train_models.py)).
