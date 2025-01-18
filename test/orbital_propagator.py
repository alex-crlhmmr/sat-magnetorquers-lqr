# orbital_propagator.py

import numpy as np

def gravitational_acceleration(r, orbit_params):
    """
    Calculates gravitational acceleration with J2 perturbation.

    Parameters:
    r (numpy.ndarray): Position vector in ECI frame (meters).
    orbit_params (dict): Dictionary containing orbital parameters.

    Returns:
    numpy.ndarray: Acceleration vector in m/s^2.
    """
    mu = orbit_params['mu']        # Earth's gravitational parameter, m^3/s^2
    J2 = orbit_params['J2']        # Earth's J2 coefficient
    R_e = orbit_params['R_e']      # Earth's radius in meters

    norm_r = np.linalg.norm(r)
    x, y, z = r
    z2 = z * z
    r2 = norm_r ** 2
    r5 = norm_r ** 5
    factor = (3/2) * J2 * mu * R_e**2 / r5

    ax = -mu * x / r2 * (1 - factor * (5 * z2 / r2 - 1))
    ay = -mu * y / r2 * (1 - factor * (5 * z2 / r2 - 1))
    az = -mu * z / r2 * (1 - factor * (5 * z2 / r2 - 3))

    return np.array([ax, ay, az])

def rk4_step(state, delta_t, orbit_params):
    """
    Performs a single RK4 integration step for orbital propagation.

    Parameters:
    state (numpy.ndarray): Current state vector [x, y, z, vx, vy, vz] (meters and m/s).
    delta_t (float): Time step in seconds.
    orbit_params (dict): Dictionary containing orbital parameters.

    Returns:
    numpy.ndarray: Updated state vector after delta_t.
    """
    def derivatives(state):
        r = state[:3]
        v = state[3:]
        a = gravitational_acceleration(r, orbit_params)
        return np.hstack((v, a))
    
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * delta_t * k1)
    k3 = derivatives(state + 0.5 * delta_t * k2)
    k4 = derivatives(state + delta_t * k3)
    
    new_state = state + (delta_t / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return new_state
