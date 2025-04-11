"""
Handle models predictions and output tab form
"""

# server/prediction_handler_simple.py

import numpy as np

def get_tab_output(softmax_output: np.ndarray, num_strings: int = 6) -> list[int]:
    """
    Converts a softmax output vector directly into a one-hot guitar tab format
    based *only* on the highest probability (argmax).

    No confidence thresholding or temporal smoothing is applied.

    Args:
        softmax_output (np.ndarray): A numpy array of probabilities from the
                                     softmax layer (length should be num_strings + 1, e.g., 7).
        num_strings (int): The number of guitar strings (should be 6).

    Returns:
        list[int]: A list of length `num_strings` (6). It's one-hot encoded
                   (e.g., [0, 1, 0, 0, 0, 0]) if the highest probability
                   corresponds to the second string (index 1), or all zeros
                   if the highest probability corresponds to the negative class
                   (index 6). Returns None on invalid input shape.
    """
    if not isinstance(softmax_output, np.ndarray):
        softmax_output = np.array(softmax_output)
    if softmax_output.shape == (1, num_strings + 1):
        softmax_output = softmax_output[0]
    # Basic validation for input shape
    if len(softmax_output) != num_strings + 1:
        print(f"Error: Expected softmax output length {num_strings + 1}, but got {len(softmax_output)}")
        return None # Or return [0] * num_strings based on desired error handling

    # 1. Find the index of the highest probability
    predicted_index = np.argmax(softmax_output)

    # 2. Create the output tab
    output_tab = [0] * num_strings

    # 3. Check if the prediction corresponds to a string (index 0 to num_strings-1)
    if 0 <= predicted_index < num_strings:
        # It's one of the strings, set the corresponding element to 1
        output_tab[predicted_index] = 1
    # else:
        # If predicted_index is num_strings (e.g., 6), it's the negative class.
        # The output_tab remains all zeros, which is correct.

    return output_tab

# --- Example Usage ---
if __name__ == '__main__':
    # Assume class mapping: 0:E4, 1:B3, 2:G3, 3:D3, 4:A2, 5:E2, 6:Negative
    pass