# src/model/prediction_handler.py

import numpy as np
import logging # Optional: for logging warnings/errors

# --- Configuration ---
NUM_STRINGS = 6
# Define the mapping from the model's output index (0-5) to the *actual string name*
# Based on user's corrected mapping:
MODEL_INDEX_TO_STRING_NAME = {
    0: "A2",
    1: "B3",
    2: "D3",
    3: "E2", # Low E
    4: "G3",
    5: "E4"  # High E
}

# Define the desired order for the final 6-element output tab array
# Order: [E4, B3, G3, D3, A2, E2]
OUTPUT_TAB_ORDER = ["E4", "B3", "G3", "D3", "A2", "E2"]

# Create a reverse mapping from string name to the desired output tab index
# This is useful for quickly finding where to put the '1'
STRING_NAME_TO_OUTPUT_INDEX = {name: idx for idx, name in enumerate(OUTPUT_TAB_ORDER)}
# ---

def get_tab_output(softmax_output: np.ndarray) -> list[int]:
    """
    Converts a softmax output vector into a one-hot guitar tab format
    based *only* on the highest probability (argmax), applying the correct
    mapping from model output index to the desired tab order.

    Desired Tab Order: [E4, B3, G3, D3, A2, E2]

    Args:
        softmax_output (np.ndarray): A numpy array of probabilities from the
                                     softmax layer (length should be NUM_STRINGS + 1, i.e., 7).

    Returns:
        list[int]: A list of length NUM_STRINGS (6) in the desired tab order.
                   It's one-hot encoded or all zeros if the negative class is predicted
                   or if the input is invalid. Returns None on critical input errors.
    """
    # --- Input Validation and Normalization ---
    if not isinstance(softmax_output, np.ndarray):
        try:
            softmax_output = np.array(softmax_output, dtype=np.float32)
        except Exception as e:
            logging.error(f"Prediction Handler: Error converting input to NumPy array: {e}")
            return None # Cannot proceed

    # Handle potential extra batch dimension (e.g., shape (1, 7))
    if softmax_output.ndim == 2 and softmax_output.shape[0] == 1:
        softmax_output = softmax_output[0]

    # Check final shape
    expected_len = NUM_STRINGS + 1
    if softmax_output.shape != (expected_len,):
        logging.error(f"Prediction Handler: Expected softmax output shape ({expected_len},), but got {softmax_output.shape}")
        # Return default all zeros for recoverable error
        return [0] * NUM_STRINGS
    # ---

    # 1. Find the index of the highest probability
    predicted_model_index = np.argmax(softmax_output)

    # 2. Initialize the output tab (all zeros)
    output_tab = [0] * NUM_STRINGS

    # 3. Check if the prediction is one of the strings (not the negative class)
    if predicted_model_index in MODEL_INDEX_TO_STRING_NAME:
        # It's a string prediction
        predicted_string_name = MODEL_INDEX_TO_STRING_NAME[predicted_model_index]

        # Find the correct index in the output_tab based on the desired order
        if predicted_string_name in STRING_NAME_TO_OUTPUT_INDEX:
            output_index = STRING_NAME_TO_OUTPUT_INDEX[predicted_string_name]
            output_tab[output_index] = 1
        else:
            # Should not happen if mappings are correct, but good to check
            logging.warning(f"Prediction Handler: Predicted string name '{predicted_string_name}' not found in desired output order.")
            # Output remains all zeros

    # else:
        # If predicted_model_index is the negative class index (e.g., 6),
        # or if it was an unexpected index not in our mapping,
        # the output_tab remains all zeros, which is correct.

    return output_tab

# --- Example Usage (for testing this file directly) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("--- Simple Handler Simulation with Correct Mapping ---")
    print(f"Desired Output Order: {OUTPUT_TAB_ORDER}")
    print(f"Model Index -> String -> Output Index:")
    for model_idx, name in MODEL_INDEX_TO_STRING_NAME.items():
         output_idx = STRING_NAME_TO_OUTPUT_INDEX.get(name, "N/A")
         print(f"  {model_idx} -> {name} -> {output_idx}")

    # Example 1: Model predicts index 5 (E4) strongly
    # Softmax: [A2, B3, D3, E2, G3, E4, Neg]
    pred1 = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.9, 0.05])
    tab1 = get_tab_output(pred1)
    print(f"\nInput 1 (Model predicts index 5 = E4): {np.round(pred1, 2)}, Output Tab: {tab1}")
    # Expected Output: [1, 0, 0, 0, 0, 0] (E4 is at index 0 in output_tab)

    # Example 2: Model predicts index 0 (A2) strongly
    pred2 = np.array([0.85, 0.01, 0.01, 0.05, 0.01, 0.01, 0.06])
    tab2 = get_tab_output(pred2)
    print(f"Input 2 (Model predicts index 0 = A2): {np.round(pred2, 2)}, Output Tab: {tab2}")
    # Expected Output: [0, 0, 0, 0, 1, 0] (A2 is at index 4 in output_tab)

    # Example 3: Model predicts index 3 (E2) strongly
    pred3 = np.array([0.01, 0.02, 0.02, 0.8, 0.05, 0.05, 0.05])
    tab3 = get_tab_output(pred3)
    print(f"Input 3 (Model predicts index 3 = E2): {np.round(pred3, 2)}, Output Tab: {tab3}")
    # Expected Output: [0, 0, 0, 0, 0, 1] (E2 is at index 5 in output_tab)

    # Example 4: Model predicts negative class (index 6)
    pred4 = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.94])
    tab4 = get_tab_output(pred4)
    print(f"Input 4 (Model predicts index 6 = Neg): {np.round(pred4, 2)}, Output Tab: {tab4}")
    # Expected Output: [0, 0, 0, 0, 0, 0]

    # Example 5: Invalid input shape
    pred5 = np.array([0.1, 0.9]) # Too short
    tab5 = get_tab_output(pred5)
    print(f"Input 5 (Invalid Shape): {np.round(pred5, 2)}, Output Tab: {tab5}")
    # Expected Output: [0, 0, 0, 0, 0, 0] (and an error logged)