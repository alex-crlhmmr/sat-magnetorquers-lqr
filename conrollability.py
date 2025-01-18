import numpy as np
from scipy.integrate import quad
from magnetic_field import magnetic_field
from datetime import datetime
from orbital import *
from scipy.integrate import solve_ivp
from controllability.orbital_propagator import *

def skew_symmetric_matrix(B):
    """
    Generate the skew-symmetric matrix for the cross-product with vector B.
    """
    return np.array([
        [0, -B[2], B[1]],
        [B[2], 0, -B[0]],
        [-B[1], B[0], 0]
    ])

def controllability_grammian_incremental(B_func, t0, tf, step=1):
    """
    Incrementally compute the controllability Grammian over [t0, t] and check the condition number.

    Parameters:
    B_func : function
        A function that returns the magnetic field vector B(t) at time t.
    t0 : float
        Start of the time interval.
    tf : float
        End of the time interval.
    step : float
        Time step for incremental integration (in seconds).

    Returns:
    t_condition_met : float
        The time when the controllability Grammian first achieves a condition number of 1.
    """
    # Initialize the Grammian as a 3x3 zero matrix
    C = np.zeros((3, 3))
    
    # Incrementally integrate from t0 to tf
    t = t0
    while t <= tf:
        # Integrate the contributions from [t, t + step]
        def integrand(t):
            B = B_func(t)
            print("Magnetic Field Vector:", B)  #
            B_cross = skew_symmetric_matrix(B)
            return B_cross @ B_cross.T

        for i in range(3):
            for j in range(3):
                integrand_ij = lambda t: integrand(t)[i, j]
                delta_C, _ = quad(integrand_ij, t, t + step)
                C[i, j] += delta_C
        
        # Compute the condition number
        cond_number = np.linalg.cond(C)
        
        # Check if the condition number is close to 1
        if np.isclose(cond_number, 1, atol=1e-3):
            return t
        
        t += step

    # If no such time is found, return None
    return None

if __name__ == "__main__":
    # Initial setup
    date = datetime(2025, 1, 18)  # Start date
    t_0 = 0  # Start time
    t_f = 1200  # End time (in seconds)
    dt = 10  # Propagation time step (seconds)

    # Example initial states
    initial_orbital_state_mee = np.array([7000, 0.01, 0.01, 0.01, 0.01, 0.01])  # Semi-parameter, eccentricity, inclination, RAAN, argument of latitude, true longitude
    initial_attitude_state = np.array([0.5, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01])  # Quaternion + angular rates

    print(mee_to_lla(initial_orbital_state_mee))




    # Initialize time and controllability Grammian
    t = t_0
    current_state_mee = initial_orbital_state_mee

    # While loop to propagate the orbital elements and calculate controllability Grammian
    while t < t_f:
        # Convert current MEE to Latitude, Longitude, Altitude
        lat,lon,alt = mee_to_lla(current_state_mee)

        # Get the magnetic field at the current position
        current_mag_field = magnetic_field(lat,lon,alt, date)
        print(current_mag_field)

        # Define a magnetic field function for incremental Grammian evaluation
        def B_func(dt):
            print("Current MEE:", current_state_mee)
            next_mee = propagate_orbit(current_state_mee, t, dt)
            print("Next MEE:", next_mee)
            lat,lon,alt = mee_to_lla(next_mee)
            result = magnetic_field(lat,lon,alt, date)
            print(result)
            return result

        # Incrementally compute the controllability Grammian
        t_condition_met = controllability_grammian_incremental(B_func, t_0, t + dt, step=1)

        # If condition number of 1 is achieved, break
        if t_condition_met is not None:
            print(f"The controllability Grammian first achieves a condition number of 1 at t = {t_condition_met} seconds.")
            print(f"Orbital state (MEE): {current_state_mee}")
            print(f"Magnetic field: {current_mag_field}")
            break

        # Propagate the orbital state for the next time step
        current_state_mee = mee_dynamics(current_state_mee, dt)
        t += dt

    if t >= t_f:
        print("The controllability Grammian does not achieve a condition number of 1 within the given interval.")
