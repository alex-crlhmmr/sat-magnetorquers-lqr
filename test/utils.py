# utils.py

import numpy as np

def skew_symmetric(B):
    """
    Constructs a skew-symmetric matrix from a 3-element vector.

    Parameters:
    B (array-like): 3-element magnetic field vector.

    Returns:
    numpy.ndarray: 3x3 skew-symmetric matrix.
    """
    return np.array([
        [0, -B[2], B[1]],
        [B[2], 0, -B[0]],
        [-B[1], B[0], 0]
    ])
