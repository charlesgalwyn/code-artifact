import numpy as np

def preprocess_input(sequence, seq_len, n_numeric):
    """
    sequence: list of timesteps
    Each timestep = {
        "action": int,
        "reg": int,
        "self": int,
        "numeric": [float, float, ...]
    }
    """

    # Pad or trim sequence
    if len(sequence) < seq_len:
        pad_len = seq_len - len(sequence)
        pad_step = {
            "action": 0,
            "reg": 0,
            "self": 0,
            "numeric": [0.0] * n_numeric
        }
        sequence = [pad_step] * pad_len + sequence
    else:
        sequence = sequence[-seq_len:]

    X_action = np.array([[step["action"] for step in sequence]])
    X_reg    = np.array([[step["reg"] for step in sequence]])
    X_self   = np.array([[step["self"] for step in sequence]])
    X_num    = np.array([[step["numeric"] for step in sequence]])

    return [X_action, X_reg, X_self, X_num]
