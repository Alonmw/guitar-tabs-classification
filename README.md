# Guitar Tabs Classification

## Overview

This project is an end-to-end system for classifying guitar tabs from audio data. It was built entirely from scratch—including data collection, preprocessing, model design, and training. The current implementation covers data processing and model training using Python. **Future work** includes developing a real-time web application (using Flask and WebSockets) to serve live audio predictions.

## Project Structure

```
/guitar-tabs-classification
├── .idea/                     # IDE configuration files (PyCharm, etc.)
├── data/                      # Collected raw and preprocessed audio data
├── models/                    # Saved model checkpoints and final models
├── notebooks/                 # Jupyter notebooks for data exploration and experimentation
├── server/                    # Backend server code for the future real-time web app
│   └── app.py                 # Flask application to serve the model & handle WebSocket connections
└── src/                       # Source code for model development and data processing
    ├── data_loader.py         # Script for loading and managing the audio dataset
    ├── preprocessing.py       # Audio preprocessing and feature extraction routines
    ├── model.py               # Definition of the machine learning model architecture
    ├── train.py               # Training pipeline for model development
    └── visualization.py       # Tools for visualizing data and model performance
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation (this file)
```

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Alonmw/guitar-tabs-classification.git
   cd guitar-tabs-classification
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

- **Data Loading & Preprocessing**

  Run the data loader and preprocessing scripts (customize as needed):

  ```bash
  python src/data_loader.py
  python src/preprocessing.py
  ```

- **Model Training**

  Train the model by running:

  ```bash
  python src/train.py
  ```

- **Evaluation**

  Use the notebooks in the `/notebooks` folder for exploratory data analysis, model evaluation, and visualization:

  ```bash
  jupyter notebook
  ```

## Future Work

- **Real-Time Web Application**\
  The next phase involves integrating the trained model into a web application for real-time predictions:

  - **Backend:** Develop a Flask server (code will reside in the `/server` directory) to process live audio, perform necessary preprocessing, and return model predictions.
  - **Frontend:** Create an interactive web UI (using React or plain JavaScript) that captures audio from the user's microphone, sends it to the backend via WebSockets, and displays live classification results.

- **Optimization for Edge Devices**\
  Explore TensorFlow Lite or other model optimization techniques for deployment on low-power devices.

## Contributing

Contributions are welcome! Please fork the repository, make your improvements, and submit a pull request. All contributions—from bug fixes to feature enhancements—are appreciated.

## License

This project is licensed under the [MIT License](LICENSE).

---

> **Note:** This project is a work in progress. While the core model and training pipeline are fully functional, the real-time web integration is still under development. Check back for updates!

