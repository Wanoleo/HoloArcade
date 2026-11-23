# HoloArcade — Webcam-Based Gesture-Controlled Game Suite

HoloArcade is a desktop game suite controlled through real-time hand gestures. It combines webcam input, MediaPipe hand-landmark detection, a trained scikit-learn gesture classifier, and Pygame-rendered interactive experiences.

The application includes three interactive experiences—**HoloEscape**, **HoloSoccer**, and **HoloBoard**—plus an in-app **ML Evaluation Report**. Keyboard controls are available for several actions.

> **Scope:** This is a webcam-based computer-vision game, not headset-based augmented reality or spatial tracking. A functioning webcam is required for the current application loop.

## Experiences

| Mode | Description | Main implementation |
| --- | --- | --- |
| **HoloEscape** | First-person sci-fi escape experience with puzzles, exploration, and narration | `src/game_logic.py`, `src/raycaster.py`, `src/renderer.py`, `src/narrator.py` |
| **HoloSoccer** | Gesture-controlled football experience | `src/soccer_game.py`, `src/renderer.py` |
| **HoloBoard** | Air-drawing canvas, virtual keyboard, and text-to-speech action | `src/board_game.py`, `src/renderer.py` |
| **ML Report** | Interactive view of model evaluation, per-class results, and runtime metrics; supports HTML export | `src/report_view.py` |

## How It Works

```text
Webcam frame
    ↓
OpenCV frame capture
    ↓
MediaPipe hand detection (21 landmarks per detected hand)
    ↓
Geometric feature normalization
    ↓
scikit-learn gesture classification
    ↓
Prediction smoothing and gesture triggers
    ↓
Pygame game state, interaction, and rendering
```

The six trained gesture classes are `open_palm`, `fist`, `pointing`, `thumbs_up`, `pinch`, and `two_fingers`. Their effects depend on the active mode. The implementation supports predictions for two hands.

## Technology

- **Language:** Python
- **Camera and image handling:** OpenCV
- **Hand tracking:** MediaPipe Hand Landmarker
- **Machine learning:** scikit-learn (KNN, SVM, Random Forest), NumPy, pandas, joblib
- **Game display and input:** Pygame (`pygame-ce` package)
- **Evaluation plots:** Matplotlib
- **Local assets:** CSV datasets, saved `.joblib` models, PNG images, JSON evaluation files

This is a **local desktop application**. It does not require a web frontend, backend server, cloud API, or database to run the committed game code.

## Getting Started (Windows PowerShell)

**Requirements:** Windows computer, working webcam, Git, and a Python installation compatible with the pinned dependencies. The repository's `requirements.txt` documents a Windows/Python 3.14.8 environment; other Python versions have not been verified here.

```powershell
git clone https://github.com/Wanoleo/HoloArcade.git
cd HoloArcade
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

If you already downloaded the source, open a terminal in its root folder and skip the `git clone` and `cd` steps. Windows may request webcam permission for the program.

The repository also provides `run_holoarcade.bat` as a Windows launcher.

### Launch a specific screen

```powershell
.\.venv\Scripts\python.exe main.py --game hub
.\.venv\Scripts\python.exe main.py --game escape
.\.venv\Scripts\python.exe main.py --game soccer
.\.venv\Scripts\python.exe main.py --game board
.\.venv\Scripts\python.exe main.py --game report
```

Optional arguments:

```powershell
.\.venv\Scripts\python.exe main.py --camera 1
.\.venv\Scripts\python.exe main.py --model models\svm_model.joblib
.\.venv\Scripts\python.exe main.py --fullscreen
```

### Keyboard navigation

| Key | Function |
| --- | --- |
| `1`, `2`, `3` | Select Escape, Soccer, or Board from the hub |
| `4` or `E` | Open the ML Report from the hub |
| `M` | Return to the main hub |
| `R` | Restart the active game or refresh the report |
| `F11` | Toggle fullscreen |
| `Q` or `Esc` | Exit application |

Within HoloEscape, `W/A/S/D` move, arrow keys turn, and `Space` or `F` interact. Within HoloSoccer, `W/A/S/D` or arrow keys move, `Space` shoots, and `E` passes. Some keys have different meanings depending on the screen.

## Machine Learning Pipeline

1. **Collect:** Capture labeled hand-landmark samples with session and participant identifiers in `dataset/raw/`.
2. **Preprocess:** Normalize geometric landmark features and split the processed samples by session to reduce leakage across nearby frames.
3. **Train:** Compare KNN, RBF SVM, and Random Forest classifiers.
4. **Evaluate:** Generate model comparisons, classification reports, confusion matrices, and runtime benchmarks.
5. **Infer:** Load a saved classifier and convert smoothed live predictions into in-game interactions.

The saved model artifacts, dataset files, and evaluation outputs are already included in this repository; retraining is not required for a normal first launch.

### Committed evaluation results

The following values are taken from [`results/metrics/model_comparison.csv`](results/metrics/model_comparison.csv):

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| KNN | 92.02% | 92.11% | 91.84% | 91.70% |
| SVM | 92.89% | 93.28% | 92.95% | 92.86% |
| **Random Forest** | **94.09%** | **94.36%** | **94.22%** | **94.10%** |

Random Forest is the selected model according to [`models/model_metadata.json`](models/model_metadata.json). These are recorded evaluation results, not a guarantee of accuracy for every user, camera, lighting condition, or background. For the evaluation details, see [`results/metrics/`](results/metrics/) and [`results/plots/`](results/plots/).

### Reproduce the ML workflow

```powershell
.\.venv\Scripts\python.exe -m src.collect_data --participant user1 --session s1
.\.venv\Scripts\python.exe -m src.train_models
.\.venv\Scripts\python.exe -m src.evaluate_models
.\.venv\Scripts\python.exe -m src.benchmark --frames 60
```

**Note:** Retraining can replace the committed model artifacts and evaluation outputs. Use a separate working branch or back them up if you want to preserve the supplied baseline.

## Repository Structure

```text
HoloArcade/
├── main.py                    # Application entry point and game hub
├── run_holoarcade.bat         # Windows launcher
├── requirements.txt          # Python dependencies
├── dataset/
│   ├── raw/                  # Gesture sessions s1–s6
│   └── processed/            # Train/test CSVs
├── models/                   # Gesture classifier artifacts and MediaPipe model
├── assets/images/            # Game background images
├── results/
│   ├── metrics/              # Evaluation and benchmark data
│   └── plots/                # Confusion matrices and comparison plot
├── src/
│   ├── camera.py             # Webcam capture
│   ├── hand_detection.py     # Hand landmark tracking
│   ├── preprocessing.py      # Feature preparation
│   ├── collect_data.py       # Gesture data collector
│   ├── train_models.py       # Model training
│   ├── evaluate_models.py    # Evaluation
│   ├── gesture_recognition.py # Real-time gesture inference
│   ├── game_logic.py         # Escape-room state and interactions
│   ├── raycaster.py          # First-person scene rendering logic
│   ├── soccer_game.py        # Soccer state and controls
│   ├── board_game.py         # Air drawing and virtual keyboard
│   ├── report_view.py        # Evaluation report screen and export
│   ├── renderer.py           # Pygame visual rendering
│   ├── narrator.py           # Speech/narration
│   ├── benchmark.py          # Performance measurement
│   └── config.py             # Shared paths and gesture definitions
└── tests/                    # Automated test modules M1–M10
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

This command runs the committed test suite. It does not replace an actual end-to-end test with a working webcam on the target machine.
## 🎮 Project Demo

Watch the HoloArcade demonstration:

[📥 Download HoloArcade Demo Video](https://github.com/Wanoleo/HoloArcade/raw/refs/heads/main/demo_video.mp4)

## Limitations and Responsible Use

- Gesture recognition performance depends on lighting, hand visibility, camera quality, and user movement.
- Some interactions also support keyboard input; the experience is not guaranteed to be fully hands-free.
- The included dataset contains gesture-landmark observations. Confirm participant consent and permissions before redistributing collected data or derivative datasets.
- No generalization claim is made beyond the committed evaluation setup.

## Contributors and License

Developed as a university machine-learning project. Collaborator names and individual contributions should be added after the team confirms how it wants to be credited.

**License:** No open-source license is included at present. Public visibility alone does not grant permission to reuse or redistribute the code. Repository owners should choose a license before inviting outside reuse.

