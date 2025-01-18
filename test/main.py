# main.py

import numpy as np
from datetime import datetime, timezone
from astropy.time import Time
from astropy import units as u

from orbital_propagator import gravitational_acceleration, rk4_step as orbit_rk4_step
from magnetic_field import get_magnetic_field_ECI
from utils import skew_symmetric
from controllability import compute_condition_number

def eci_to_geodetic(r_ECI, current_time_astropy):
    """
    Converts ECI coordinates to geodetic coordinates.

    Parameters:
    r_ECI (numpy.ndarray): Position vector in ECI frame (meters).
    current_time_astropy (astropy.time.Time): Current simulation time.

    Returns:
    tuple: (latitude in degrees, longitude in degrees, altitude in meters)
    """
    from astropy.coordinates import TEME, ITRS, EarthLocation, SkyCoord

    teme = SkyCoord(
        x=r_ECI[0] * u.m,
        y=r_ECI[1] * u.m,
        z=r_ECI[2] * u.m,
        frame='teme',
        obstime=current_time_astropy,
        representation_type='cartesian'
    )

    itrs = teme.transform_to('itrs')
    location = EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z, unit=u.m)
    geodetic = location.geodetic

    lat = geodetic.lat.degree
    lon = geodetic.lon.degree
    alt = geodetic.height.to(u.m).value

    return lat, lon, alt

def compute_maneuver_duration(orbit_params, delta_t, cutoff_condition_number, max_time, initial_time_astropy, initial_state):
    """
    Computes the maneuver duration tf using RK4 integration for state and Euler for C.

    Parameters:
    orbit_params (dict): Dictionary containing orbital parameters.
    delta_t (float): Time step in seconds.
    cutoff_condition_number (float): Condition number threshold.
    max_time (float): Maximum allowed maneuver time in seconds.
    initial_time_astropy (astropy.time.Time): Initial simulation time.
    initial_state (numpy.ndarray): Initial state vector [x, y, z, vx, vy, vz] in meters and m/s.

    Returns:
    float or None: Maneuver duration tf in seconds if achieved, else None.
    """
    # Initialize Controllability Grammian C as a zero matrix
    C = np.zeros((3,3))
    
    # Initialize state
    state = initial_state.copy()
    
    # Initialize time
    t = 0.0  # seconds
    
    # Simulation loop
    while t < max_time:
        # Current simulation time
        current_time_astropy = initial_time_astropy + t * u.s
        # FIX: Produce timezone-naive datetime
        sim_datetime_py = current_time_astropy.to_datetime()  # Removed timezone=timezone.utc
        
        # Satellite position
        r_ECI = state[:3]  # meters
        
        # Convert ECI to geodetic coordinates
        lat, lon, alt = eci_to_geodetic(r_ECI, current_time_astropy)
        
        # Convert to spherical coordinates for IGRF
        r_km = np.linalg.norm(r_ECI) / 1e3  # km
        theta_deg = 90.0 - lat  # colatitude
        phi_deg = lon % 360.0    # longitude [0, 360)
        
        # Get Earth's magnetic field in ECI coordinates
        B_ECI = get_magnetic_field_ECI(r_km, theta_deg, phi_deg, sim_datetime_py)  # Tesla
        
        # Construct skew-symmetric matrix Bx
        Bx = skew_symmetric(B_ECI)
        
        # Compute dC/dt = Bx * Bx
        dC_dt = np.dot(Bx, Bx)
        
        # Integrate C using Euler's method
        C += dC_dt * delta_t
        
        # Integrate satellite state using RK4
        state = orbit_rk4_step(state, delta_t, orbit_params)
        
        # Compute condition number of C
        condition_num = compute_condition_number(C)
        
        print(f"Time: {t:.1f} s, Condition Number: {condition_num:.2e}")
        
        # Check if condition number meets the cutoff
        if condition_num < cutoff_condition_number:
            print(f"Desired condition number achieved at tf = {t:.1f} seconds")
            return t
        
        # Increment time
        t += delta_t
    
    # If maximum time is reached without achieving cutoff
    print("Maximum time reached without achieving desired condition number.")
    return None

if __name__ == "__main__":
    # Define orbital parameters
    orbit_params = {
        'semi_major_axis': 6371e3 + 500e3,  # Earth's radius + 500 km, in meters
        'eccentricity': 0.0,                # Circular orbit
        'inclination': 45.0,                 # Degrees
        'raan': 0.0,                         # Right Ascension of Ascending Node (degrees)
        'arg_pe': 0.0,                       # Argument of Perigee (degrees)
        'true_anomaly_0': 0.0,               # True Anomaly at epoch (degrees)
        'mu': 3.986004418e14,                # Earth's gravitational parameter, m^3/s^2
        'J2': 1.08263e-3,                    # Earth's J2 coefficient
        'R_e': 6378137.0                      # Earth's radius in meters
    }
    
    # Define simulation parameters
    delta_t = 60.0                     # Time step in seconds
    cutoff_condition_number = 1e6      # Condition number cutoff
    max_time = 86400.0                  # Maximum maneuver time in seconds (1 day)
    
    # Define initial simulation time (UTC)
    initial_time_str = '2021-01-01T00:00:00'
    initial_time_astropy = Time(initial_time_str, scale='utc')
    
    # Define initial state [x, y, z, vx, vy, vz] in meters and m/s
    # Assuming a circular orbit in ECI frame
    a = orbit_params['semi_major_axis']
    mu = orbit_params['mu']
    v0 = np.sqrt(mu / a)  # Circular orbit velocity
    initial_state = np.array([
        orbit_params['semi_major_axis'],  # x position in meters
        0.0,                              # y position in meters
        0.0,                              # z position in meters
        0.0,                              # vx velocity in m/s
        v0,                               # vy velocity in m/s
        0.0                               # vz velocity in m/s
    ])
    
    # Compute maneuver duration tf
    tf = compute_maneuver_duration(
        orbit_params=orbit_params,
        delta_t=delta_t,
        cutoff_condition_number=cutoff_condition_number,
        max_time=max_time,
        initial_time_astropy=initial_time_astropy,
        initial_state=initial_state
    )
    
    if tf is not None:
        print(f"Maneuver duration tf = {tf:.1f} seconds")
    else:
        print("Failed to determine maneuver duration within the maximum allowed time.")