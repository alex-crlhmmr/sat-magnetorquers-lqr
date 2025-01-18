# magnetic_field.py

import numpy as np
from ppigrf import igrf_gc

def get_magnetic_field_ECI(r_km, theta_deg, phi_deg, date):
    """
    Computes the Earth's magnetic field vector in ECI Cartesian coordinates.

    Parameters:
    r_km (float): Radial distance from Earth's center in kilometers.
    theta_deg (float): Colatitude in degrees (0° at North Pole).
    phi_deg (float): Longitude in degrees east.
    date (datetime.datetime): Observation date (timezone-naive, UTC).

    Returns:
    numpy.ndarray: Magnetic field vector in ECI frame (Tesla).
    """
    # Retrieve magnetic field components from IGRF
    Br, Btheta, Bphi = igrf_gc(r_km, theta_deg, phi_deg, date)  # in nT

    # Convert degrees to radians
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)

    # Convert spherical to Cartesian coordinates
    Bx = float(Br) * np.sin(theta_rad) * np.cos(phi_rad) + \
         float(Btheta) * np.cos(theta_rad) * np.cos(phi_rad) - \
         float(Bphi) * np.sin(phi_rad)
    
    By = float(Br) * np.sin(theta_rad) * np.sin(phi_rad) + \
         float(Btheta) * np.cos(theta_rad) * np.sin(phi_rad) + \
         float(Bphi) * np.cos(phi_rad)
    
    Bz = float(Br) * np.cos(theta_rad) - float(Btheta) * np.sin(theta_rad)

    # Convert from nT to Tesla and ensure it's a 1D array of floats
    B_ECI = np.array([Bx, By, Bz], dtype=float) * 1e-9  # Tesla

    # Debugging: Print the shape and contents of B_ECI
    # Uncomment the following line if you need to debug
    # print(f"B_ECI: {B_ECI}, Shape: {B_ECI.shape}")

    return B_ECI