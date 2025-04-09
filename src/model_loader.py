import logging
from keras import models
def load_trained_model(path: str):
    try:
        model = models.load_model(path)
        logging.info(f"Loaded model from {path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise