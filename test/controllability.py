# controllability.py

import numpy as np
from numpy.linalg import cond

def compute_condition_number(C):
    """
    Computes the condition number of the Controllability Grammian matrix.

    Parameters:
    C (numpy.ndarray): 3x3 Controllability Grammian matrix.

    Returns:
    float: Condition number.
    """
    return cond(C)
