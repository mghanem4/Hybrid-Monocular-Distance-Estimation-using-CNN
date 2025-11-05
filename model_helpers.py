import numpy as np

def ensure_P2_matrix(P2: np.ndarray) -> np.ndarray:
    """
    Your calib dict value is a flat vector of 12 elements.
    Convert to (3,4) if needed.
    """
    P2 = np.asarray(P2)
    if P2.shape == (12,):
        return P2.reshape(3, 4)
    if P2.shape == (1, 12):  # in case it's wrapped once
        return P2.reshape(3, 4)
    if P2.shape == (3, 4):
        return P2
    raise ValueError(f"Unexpected P2 shape {P2.shape}; expected 12 or (3,4).")
