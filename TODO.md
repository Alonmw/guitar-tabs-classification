# TODO List - Guitar Tabs Real-time Classifier App

This list outlines planned improvements and remaining tasks for the project.

## Performance & Latency (High Priority)

- [ ] **Reduce Audio Window Size:**
    - [ ] Decide on an optimal shorter duration (target: 250ms-500ms).
    - [ ] Update `WINDOW_SIZE` constant in `server/audio_buffer.py`.
    - [ ] Adjust related buffer logic if needed.
- [ ] **Reformat Training Data:**
    - [ ] Use data extraction scripts (`extract_onset_chunks.py`, `extract_multi_onset_chunks.py`) to create datasets based on the new shorter window size. Ensure sufficient and diverse data, especially for the negative class.
- [ ] **Adapt Preprocessing:**
    - [ ] In `src/preprocessing.py` / `server/audio_prep.py`: Adjust CQT parameters for the shorter window OR switch to Mel Spectrograms (recommended).
    * [ ] Ensure `preprocess_buffer` function works correctly with the new features/window size.
- [ ] **Retrain Model:**
    - [ ] Train a new model version from scratch using the reformatted data and updated preprocessing features.
    * [ ] Ensure the model's input layer matches the new feature shape.
- [ ] **Update Runtime Logic:**
    - [ ] Modify `server/audio_processor.py` to correctly call the updated `preprocess_buffer` and reshape data for the *new* model's input.
    - [ ] Update the `MODEL_FILENAME` in `server/app.py` (or config) to point to the newly trained model.

## Prediction Stability & Accuracy

- [ ] **Implement Robust `PredictionHandler`:**
    - [ ] Create/Use the class-based `PredictionHandler` (from earlier discussion) with state management.
    * [ ] Implement confidence threshold checking.
    * [ ] Implement temporal smoothing (e.g., persistence/debouncing counter).
    - [ ] Replace the `prediction_handler.get_tab_output` call in `server/audio_processor.py` with calls to an instance of the new handler class.
    - [ ] Initialize the handler instance in `server/app.py` and pass it to the background thread.
- [ ] **Tune Handler Parameters:**
    - [ ] Experiment with `confidence_threshold` and `persistence_threshold` values to find a good balance between responsiveness and stability for the UI output.
- [ ] **Refine Class Mapping:**
    - [ ] Double-check and confirm the `MODEL_INDEX_TO_STRING_NAME` and `OUTPUT_TAB_ORDER` mappings used in `src/model/prediction_handler.py` are correct, especially after retraining.

## Model Development & Evaluation (*New Section*)

- [ ] **Create/Maintain EDA & Evaluation Notebook:**
    - [ ] Structure a Jupyter Notebook (`notebooks/EDA_and_Evaluation.ipynb`) for analysis.
    - [ ] Load processed data chunks (using `src/data_utils/data_loader.py` if applicable).
    - [ ] Perform EDA: Visualize features (spectrograms), check class balance, analyze audio properties of chunks.
    - [ ] Load specific trained model checkpoints (`models/`).
    - [ ] Evaluate on a dedicated test set: Calculate overall accuracy, precision, recall, F1-score (especially **per-class** metrics).
    - [ ] Generate and visualize a Confusion Matrix to understand misclassifications.
    - [ ] Implement functionality to inspect examples the model gets wrong (e.g., listen to audio, view spectrogram).
    - [ ] Keep the notebook organized for reusability with different datasets/models.

## Web Application Features & UI

- [ ] **UI Enhancements:**
    - [ ] Improve visual styling of string indicators in `index.html` / `style.css`.
    - [ ] Add clearer visual feedback for "Connected", "Disconnected", "Processing", or "Error" states.
- [ ] **Display Confidence:** (Optional) Modify `emit_prediction_updates` in `app.py` and the handler in `main.js` to display the confidence score of the stable prediction.
- [ ] **Error Display:** Implement sending error messages (e.g., audio device failure, model issues) from the backend via SocketIO to be displayed cleanly in the UI.

## Code Quality & Refactoring

- [ ] **Configuration File:** Create `config.py` or use `.env` / YAML to store settings like model path, thresholds, audio parameters, ports etc., instead of hardcoding them. Load these settings in `app.py`.
- [ ] **Refined Error Handling:** Add more specific `try...except` blocks and logging within the background threads (`audio_processor`, `emit_prediction_updates`).
- [ ] **Logging:** Configure Python's `logging` more formally (e.g., using named loggers per module, potentially adding file output).
- [ ] **Shutdown Logic:**
    - [ ] Ensure `audio_stream.py` has a reliable `stop_stream()` function (or rely solely on `sounddevice.stop()`).
    * [ ] Ensure `audio_buffer.py` has a reliable `stop_buffer_thread()` function.
    - [ ] Test the `atexit` shutdown sequence thoroughly by stopping the server (`Ctrl+C`). Consider using signal handling for more robustness.
- [ ] **Static Files:** Move inline CSS and JS from `index.html` to `server/static/css/style.css` and `server/static/js/main.js`. Link them using `url_for` in the template.
- [ ] **Testing:** Add unit/integration tests for key components.
- [ ] **Review Multiprocessing:** (Low Priority) After other optimizations, decide if re-introducing multiprocessing for preprocessing is necessary and feasible within the Flask/SocketIO context.

---
