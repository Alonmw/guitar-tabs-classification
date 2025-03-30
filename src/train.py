import os
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_loader import get_data_dir, get_xy  # Assuming this is your function in data_loader.py
from model import build_model  # Assuming this is your function in model.py
from src.visualization import ROOT_DIR


def load_and_split():
    # Step 1: Load preprocessed data
    print("Loading preprocessed data...")
    data_dir = get_data_dir()
    X, y = get_xy(data_dir)  # Assuming this returns features (x_data) and labels (y_data)
    print(f'X shape: {X.shape}, y shape: {y.shape}')
    # Step 2: Split the data into training, validation, and test sets
    print("Splitting data into training, validation, and test sets...")
    x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)  # 80% train, 20% temp
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)  # 10% val, 10% test

    print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}, Test samples: {len(x_test)}")
    return x_train, y_train, x_val, y_val, x_test, y_test

def train_and_save(x_train, y_train, x_val, y_val, x_test, y_test):
    # Step 3: Build the model
    input_shape = x_train[0].shape
    num_classes = len(np.unique(y_train))  # Number of unique labels/classes in your dataset
    model = build_model(input_shape, num_classes)

    # Step 4: Compile the model
    print("Compiling the model...")
    model.compile(optimizer='adam',  # You can choose a different optimizer
                  loss='sparse_categorical_crossentropy',  # Use sparse if labels are integers
                  metrics=['accuracy'])

    # Step 6: Train the model
    print("Starting training...")
    batch_size = 1
    epochs = 5  # Adjust based on your dataset size and performance

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
    )

    # Step 7: Evaluate the model on the test set
    print("Evaluating the model on the test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Save the final model
    final_model_path = ROOT_DIR + '/models/model.keras'
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")


if __name__ == '__main__':
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_split()
    train_and_save(x_train, y_train, x_val, y_val, x_test, y_test)