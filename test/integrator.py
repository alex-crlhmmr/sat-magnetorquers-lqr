# integrator.py

import numpy as np

def rk4_step(state, delta_t, derivatives_func):
    """
    Performs a single RK4 integration step for a generic state.

    Parameters:
    state (numpy.ndarray): Current state vector.
    delta_t (float): Time step in seconds.
    derivatives_func (callable): Function that returns derivatives given state and time.

    Returns:
    numpy.ndarray: Updated state vector after delta_t.
    """
    k1 = derivatives_func(state)
    k2 = derivatives_func(state + 0.5 * delta_t * k1)
    k3 = derivatives_func(state + 0.5 * delta_t * k2)
    k4 = derivatives_func(state + delta_t * k3)
    
    new_state = state + (delta_t / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return new_state
