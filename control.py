# Import necessary libraries
import numpy as np
from ppigrf import igrf_gc
from datetime import datetime, timezone
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import TEME, ITRS, EarthLocation, CartesianRepresentation, SkyCoord
from numpy.linalg import cond
import warnings

# Suppress warnings from ppigrf if any (optional)
warnings.filterwarnings("ignore")

# Define function to compute skew-symmetric matrix
def skew_symmetric(B):
    """
    Returns the skew-symmetric matrix of vector B.
    B: array-like, shape (3,)
    """
    return np.array([
        [0, -B[2], B[1]],
        [B[2], 0, -B[0]],
        [-B[1], B[0], 0]
    ])

# Define function to compute derivative of C
def dC_dt(Bx):
    """
    Computes the derivative of C at time t.
    Bx: skew-symmetric matrix, shape (3,3)
    Returns: derivative dC/dt = Bx * Bx
    """
    return np.dot(Bx, Bx)

# Define function to compute satellite position in ECI frame
def get_satellite_position(t, orbit_params):
    """
    Computes the satellite's ECI position at time t.
    t: time since epoch in seconds
    orbit_params: dictionary containing orbit parameters
    Returns: position vector in ECI frame in meters
    """
    # Unpack orbit parameters
    a = orbit_params['semi_major_axis']  # meters
    e = orbit_params['eccentricity']
    i = orbit_params['inclination']  # degrees
    raan = orbit_params['raan']  # degrees
    arg_pe = orbit_params['arg_pe']  # degrees
    true_anomaly_0 = orbit_params['true_anomaly_0']  # degrees

    # Compute orbital period using Kepler's third law
    mu = orbit_params['mu']  # Earth's gravitational parameter, m^3/s^2
    T = 2 * np.pi * np.sqrt(a**3 / mu)  # seconds

    # Mean motion
    n = 2 * np.pi / T  # rad/s

    # Mean anomaly at time t
    M = np.deg2rad(true_anomaly_0) + n * t

    # Solve Kepler's Equation for E (Eccentric Anomaly)
    # For circular orbit, E = M
    E = M  # radians

    # True anomaly
    theta = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                           np.sqrt(1 - e) * np.cos(E / 2))

    # Distance (for circular orbit, r = a)
    r = a * (1 - e * np.cos(E))

    # Position in orbital plane
    x_orb = r * np.cos(theta)
    y_orb = r * np.sin(theta)
    z_orb = 0

    # Convert degrees to radians for rotation angles
    raan_rad = np.deg2rad(raan)
    i_rad = np.deg2rad(i)
    arg_pe_rad = np.deg2rad(arg_pe)

    # Rotation matrices
    R3_raan = np.array([
        [np.cos(raan_rad), -np.sin(raan_rad), 0],
        [np.sin(raan_rad), np.cos(raan_rad), 0],
        [0, 0, 1]
    ])

    R1_inc = np.array([
        [1, 0, 0],
        [0, np.cos(i_rad), -np.sin(i_rad)],
        [0, np.sin(i_rad), np.cos(i_rad)]
    ])

    R3_arg_pe = np.array([
        [np.cos(arg_pe_rad), -np.sin(arg_pe_rad), 0],
        [np.sin(arg_pe_rad), np.cos(arg_pe_rad), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix: R = R3(RAAN) * R1(inclination) * R3(arg_pe)
    rotation_matrix = R3_raan @ R1_inc @ R3_arg_pe

    # Position in orbital plane
    r_orbital = np.array([x_orb, y_orb, z_orb])

    # Position in ECI frame
    r_ECI = rotation_matrix @ r_orbital

    return r_ECI  # in meters

# Define function to convert ECI to geodetic coordinates
def eci_to_geodetic(r_ECI, current_time):
    """
    Converts ECI position to geodetic coordinates.
    r_ECI: position vector in ECI frame in meters
    current_time: astropy Time object
    Returns: latitude (degrees), longitude (degrees), altitude (meters)
    """
    # Create SkyCoord in TEME frame
    teme = SkyCoord(
        x=r_ECI[0] * u.m,
        y=r_ECI[1] * u.m,
        z=r_ECI[2] * u.m,
        frame='teme',
        obstime=current_time,
        representation_type='cartesian'
    )

    # Transform to ITRS (Earth-fixed)
    itrs = teme.transform_to('itrs')

    # Get geodetic coordinates
    location = EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z, unit=u.m)
    geodetic = location.geodetic

    lat = geodetic.lat.degree
    lon = geodetic.lon.degree
    alt = geodetic.height.to(u.m).value

    return lat, lon, alt

# Define function to get magnetic field in spherical coordinates using ppigrf
def get_magnetic_field_spherical(r_km, theta_deg, phi_deg, date):
    """
    Gets the Earth's magnetic field in spherical coordinates using ppigrf.
    r_km: radial distance from Earth's center in kilometers
    theta_deg: colatitude in degrees (0° at North Pole)
    phi_deg: longitude in degrees east
    date: datetime object (timezone-aware, UTC)
    Returns: Br, Btheta, Bphi in nanotesla (nT)
    """
    Br, Btheta, Bphi = igrf_gc(r_km, theta_deg, phi_deg, date)  # in nT
    return Br, Btheta, Bphi

# Convert spherical B to ECI Cartesian coordinates
def spherical_to_cartesian_B(Br, Btheta, Bphi, theta_deg, phi_deg):
    """
    Converts magnetic field from spherical to ECI Cartesian coordinates.
    Br: radial component in nT
    Btheta: colatitudinal component in nT (positive southward)
    Bphi: longitudinal component in nT (positive eastward)
    theta_deg: colatitude in degrees
    phi_deg: longitude in degrees east
    Returns: B_ECI in Tesla
    """
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)

    # Convert spherical to Cartesian coordinates
    # Note: Btheta is positive southward, which corresponds to the polar (theta) direction
    Bx = Br * np.sin(theta_rad) * np.cos(phi_rad) + Btheta * np.cos(theta_rad) * np.cos(phi_rad) - Bphi * np.sin(phi_rad)
    By = Br * np.sin(theta_rad) * np.sin(phi_rad) + Btheta * np.cos(theta_rad) * np.sin(phi_rad) + Bphi * np.cos(phi_rad)
    Bz = Br * np.cos(theta_rad) - Btheta * np.sin(theta_rad)

    # Convert from nT to Tesla
    B_ECI = np.array([Bx, By, Bz]) * 1e-9  # Tesla

    return B_ECI

# Main function to compute maneuver duration
def compute_maneuver_duration(orbit_params, delta_t, cutoff_condition_number, max_time, initial_time_astropy):
    """
    Computes the maneuver duration tf using Runge-Kutta integration.
    orbit_params: dictionary containing orbit parameters
    delta_t: time step in seconds
    cutoff_condition_number: condition number threshold
    max_time: maximum allowed maneuver time in seconds
    initial_time_astropy: astropy Time object representing t0
    Returns: tf in seconds
    """
    # Initialize C as zero matrix
    C = np.zeros((3,3))

    # Initialize time
    t = 0  # seconds

    # Loop until condition number < cutoff or max_time reached
    while t < max_time:
        # Current simulation time
        current_time_astropy = initial_time_astropy + t * u.s
        sim_datetime_py = current_time_astropy.to_datetime(timezone=timezone.utc)

        # Get satellite position in ECI
        r_ECI = get_satellite_position(t, orbit_params)  # in meters

        # Convert ECI to geodetic coordinates
        lat, lon, alt = eci_to_geodetic(r_ECI, current_time_astropy)

        # Convert to spherical coordinates for ppigrf
        r_km = np.linalg.norm(r_ECI) / 1e3  # Convert meters to kilometers
        theta_deg = 90.0 - lat  # colatitude
        phi_deg = lon % 360.0  # longitude wrapping

        # Get magnetic field in spherical coordinates
        Br, Btheta, Bphi = get_magnetic_field_spherical(r_km, theta_deg, phi_deg, sim_datetime_py)

        # Convert spherical B to Cartesian ECI
        B_ECI = spherical_to_cartesian_B(Br, Btheta, Bphi, theta_deg, phi_deg)

        # Construct skew-symmetric matrix Bx
        Bx = skew_symmetric(B_ECI)

        # Compute dC/dt = Bx * Bx
        dC = dC_dt(Bx)

        # Integrate using 4th-order Runge-Kutta (RK4)
        # k1 = f(t, C)
        k1 = dC

        # Estimate Bx at t + delta_t/2 for k2
        t_half = t + delta_t / 2
        current_time_half_astropy = initial_time_astropy + t_half * u.s
        sim_datetime_half_py = current_time_half_astropy.to_datetime(timezone=timezone.utc)

        # Satellite position at t + delta_t/2
        r_ECI_half = get_satellite_position(t_half, orbit_params)
        lat_half, lon_half, alt_half = eci_to_geodetic(r_ECI_half, current_time_half_astropy)

        # Spherical coordinates for half-step
        r_km_half = np.linalg.norm(r_ECI_half) / 1e3  # km
        theta_deg_half = 90.0 - lat_half
        phi_deg_half = lon_half % 360.0

        # Get magnetic field at half-step
        Br_half, Btheta_half, Bphi_half = get_magnetic_field_spherical(r_km_half, theta_deg_half, phi_deg_half, sim_datetime_half_py)

        # Convert to Cartesian ECI
        B_ECI_half = spherical_to_cartesian_B(Br_half, Btheta_half, Bphi_half, theta_deg_half, phi_deg_half)

        # Skew-symmetric matrix at half-step
        Bx_half = skew_symmetric(B_ECI_half)

        # k2 = f(t + delta_t/2, C + k1 * delta_t/2)
        k2 = dC_dt(Bx_half)

        # k3 = f(t + delta_t/2, C + k2 * delta_t/2)
        k3 = dC_dt(Bx_half)

        # Estimate Bx at t + delta_t for k4
        t_full = t + delta_t
        current_time_full_astropy = initial_time_astropy + t_full * u.s
        sim_datetime_full_py = current_time_full_astropy.to_datetime(timezone=timezone.utc)

        # Satellite position at t + delta_t
        r_ECI_full = get_satellite_position(t_full, orbit_params)
        lat_full, lon_full, alt_full = eci_to_geodetic(r_ECI_full, current_time_full_astropy)

        # Spherical coordinates for full step
        r_km_full = np.linalg.norm(r_ECI_full) / 1e3  # km
        theta_deg_full = 90.0 - lat_full
        phi_deg_full = lon_full % 360.0

        # Get magnetic field at full step
        Br_full, Btheta_full, Bphi_full = get_magnetic_field_spherical(r_km_full, theta_deg_full, phi_deg_full, sim_datetime_full_py)

        # Convert to Cartesian ECI
        B_ECI_full = spherical_to_cartesian_B(Br_full, Btheta_full, Bphi_full, theta_deg_full, phi_deg_full)

        # Skew-symmetric matrix at full step
        Bx_full = skew_symmetric(B_ECI_full)

        # k4 = f(t + delta_t, C + k3 * delta_t)
        k4 = dC_dt(Bx_full)

        # Update C
        C += (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Compute condition number
        condition_number = cond(C)

        print(f"Time: {t:.1f} s, Condition Number: {condition_number:.2e}")

        # Check condition number
        if condition_number < cutoff_condition_number:
            print(f"Desired condition number achieved at tf = {t:.1f} seconds")
            return t

        # Increment time
        t += delta_t

    print("Maximum time reached without achieving desired condition number.")
    return None

# Example usage
if __name__ == "__main__":
    # Define orbit parameters
    orbit_params = {
        'semi_major_axis': 6371e3 + 500e3,  # Earth radius + 500 km, in meters
        'eccentricity': 0.0,  # Circular orbit
        'inclination': 45.0,  # degrees
        'raan': 0.0,  # degrees
        'arg_pe': 0.0,  # degrees
        'true_anomaly_0': 0.0,  # degrees
        'mu': 3.986004418e14  # Earth's gravitational parameter, m^3/s^2
    }

    # Define simulation parameters
    delta_t = 60.0  # seconds (increased time step for efficiency)
    cutoff_condition_number = 1e6  # example cutoff
    max_time = 86400.0  # 1 day in seconds

    # Define initial time (UTC) as Astropy Time object
    initial_time_str = '2021-01-01T00:00:00'
    initial_time_astropy = Time(initial_time_str, scale='utc')

    # Compute maneuver duration
    tf = compute_maneuver_duration(orbit_params, delta_t, cutoff_condition_number, max_time, initial_time_astropy)

    if tf is not None:
        print(f"Maneuver duration tf = {tf:.1f} seconds")
    else:
        print("Failed to determine maneuver duration within maximum time.")
