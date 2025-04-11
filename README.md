# Real-time Guitar Tab Classifier (In Progress)

This project aims to create a **real-time guitar *tablature* classifier**. It captures live audio from a microphone, processes it, and uses machine learning to identify notes and chords being played on a guitar.

**Currently, the implementation focuses on classifying the 6 open strings** (E4, B3, G3, D3, A2, E2) and includes a 'negative' class for silence, noise, or other non-guitar sounds. The results for this initial stage are displayed live via a web interface built with Flask and Socket.IO.

**This project is under active development.**

## Features (Current Stage)

* Real-time audio capture from microphone using `sounddevice`.
* Background audio processing pipeline (buffering, preprocessing, inference) using Python threading.
* Keras model (`.h5`) for **currently classifying 6 open strings + negative class** (7 classes total).
* Flask web server backend.
* Real-time prediction updates pushed to the web UI using Flask-SocketIO (WebSockets).
* Simple web UI displaying the currently predicted open string highlight.
* Utility scripts (`src/data_collection_scripts/`) for preparing audio data (e.g., extracting chunks based on onsets).

## Project Structure

guitar-tabs-classification/
├── .venv/                  # Python virtual environment
├── data/                   # Directory for storing raw/processed datasets
├── models/                 # Contains trained model files (e.g., updated_model.h5)
├── notebooks/              # Jupyter notebooks for experimentation (currently outdated)
├── server/                 # Core backend application
│   ├── init.py
│   ├── app.py              # Main Flask app, SocketIO setup, routes, background task coordination
│   ├── audio_stream.py     # Handles microphone input --> audio_queue
│   ├── audio_buffer.py     # Consumes audio_queue --> provides analysis window buffer
│   ├── audio_prep.py       # Wrapper for calling preprocessing logic
│   ├── audio_processor.py  # Runs the main background audio processing loop
│   ├── static/             # CSS, client-side JS files (to be added properly)
│   └── templates/          # HTML templates for the web UI (index.html)
├── src/                    # Source code for non-server specific logic
│   ├── init.py
│   ├── data_collection_scripts/ # Scripts to prepare audio data chunks
│   │   ├── extract_multi_onset_chunks.py
│   │   └── split_wav_script.py
│   ├── data_utils/         # Data loading utilities (e.g., data_loader.py)
│   │   └── preprocessing.py  # Core preprocessing functions (e.g., CQT)
│   └── model/              # Model definition, training, handling
│       ├── init.py
│       ├── model_loader.py   # Utility to load the trained Keras model
│       └── prediction_handler.py # Converts model output (softmax) to tab format
│       └── (model.py, train.py likely here too)
├── run.py                  # Script to start the Flask/SocketIO server
└── requirements.txt        # Project dependencies

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Alonmw/guitar-tabs-classification.git
    cd guitar-tabs-classification
    ```
2.  **Create a Python virtual environment:**
    ```bash
    python -m venv .venv
    ```
3.  **Activate the virtual environment:**
    * macOS / Linux: `source .venv/bin/activate`
    * Windows (Git Bash): `source .venv/Scripts/activate`
    * Windows (CMD/PowerShell): `.\venv\Scripts\activate`
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure system dependencies like PortAudio are installed if needed for `sounddevice` on your OS).*

## Configuration

Currently, configuration variables like the model path (`models/updated_model.h5`) are primarily set within `server/app.py`. Future work includes moving these to a dedicated configuration file.

Ensure the model file specified in `server/app.py` exists in the `models/` directory.

## Running the Application (Current Open String Version)

1.  **Activate the virtual environment** (if not already active).
2.  **Ensure a microphone** is connected and configured as the default system input (or modify `audio_stream.py` if necessary to select a specific device).
3.  **Run the server:**
    ```bash
    python run.py
    ```
4.  The server will start (by default on port 5001).
5.  **Access the web UI:** Open your web browser and navigate to `http://localhost:5001` (or `http://<your-server-ip>:5001` if running on a different machine). The UI will currently show predictions for open strings.

## Current Status & Known Issues

The project is operational at its current stage (open string detection), displaying live results on the web UI. However, it is **actively being developed** towards the goal of full tab classification.

Current limitations include:

* **Latency & Responsiveness:** Significant delay (~1-2 seconds based on current 2s window) between playing a note and seeing the prediction update. Predictions are slow to change when notes change. (See Future Work).
* **Prediction Stability:** Uses a simple `argmax` handler; predictions can be jumpy or inaccurate, especially during silence or transitions. (See Future Work).
* **Scope:** Currently only detects the 6 open strings, not fretted notes or chords.

## Future Work / TODO

Key areas for improvement to reach the goal of full tab classification and improve current functionality:

* **Model Enhancements (Towards Tabs):**
    * `[ ]` Extend model capabilities to detect fretted notes (requires new data and likely model architecture changes).
    * `[ ]` Potentially add chord detection capabilities.
* **Performance/Latency:**
    * `[ ]` Reduce the audio analysis window size (e.g., to 250-500ms).
    * `[ ]` Reformat the training dataset using the onset-detection scripts into shorter chunks (include fretted notes when available).
    * `[ ]` Adapt preprocessing (e.g., Mel Spectrograms) for shorter windows.
    * `[ ]` Retrain the model(s) on the new data format.
    * `[ ]` Update the runtime audio buffer and processing logic.
* **Prediction Stability:**
    * `[ ]` Implement the more robust class-based `PredictionHandler` with confidence thresholding and temporal smoothing.
    * `[ ]` Tune the handler's parameters.
* **Model Evaluation:**
    * `[ ]` Create/maintain a Jupyter Notebook for EDA and detailed model performance evaluation (accuracy, per-class metrics, confusion matrix, error analysis).
* **Code Quality & Features:**
    * `[ ]` Move settings to a configuration file.
    * `[ ]` Improve error handling and logging in background threads.
    * `[ ]` Refine UI/UX (potentially display full tabs later) and add clearer status indicators.
    * `[ ]` Ensure robust application shutdown logic.
    * `[ ]` Move static CSS/JS to separate files.

## Dependencies

See `requirements.txt` for a full list. Key libraries include:

* Flask, Flask-SocketIO, eventlet
* TensorFlow / Keras
* librosa
* soundfile
* sounddevice
* NumPy
