from numba import jit
import numpy as np
from constants import *

@jit(nopython=True)
def mee_dynamics(elements, mu, f_perturbation):
    """Compute rigid body dynamics in Modified Equinoctial Elements (MEE).

    Args:
        elements (np.ndarray): Orbital elements [p, f, g, h, k, L].
        mu (float): Gravitational parameter.
        dt (float): Time step (unused here, but kept for consistency).
        f_perturbation (np.ndarray): Perturbation forces [fr, ft, fn].

    Returns:
        np.ndarray: Time derivative of MEE.
    """
    # Extract elements
    p, f, g, h, k, L = elements

    # Precompute reusable terms
    cosL = np.cos(L)
    sinL = np.sin(L)
    w = 1 + f * cosL + g * sinL
    root_p_mu = np.sqrt(p / mu)
    inv_w = 1 / w
    s_2 = 1 + h**2 + k**2
    w_plus_1 = w + 1

    # Preallocate matrix A for faster computation (with numba)
    A = np.zeros((6, 3))

    # Fill matrix A directly
    A[0, 1] = 2 * p * inv_w
    A[1, 0] = sinL
    A[1, 1] = inv_w * (w_plus_1 * cosL + f)
    A[1, 2] = -(g * inv_w) * (h * sinL - k * cosL)
    A[2, 0] = -cosL
    A[2, 1] = inv_w * (w_plus_1 * sinL + g)
    A[2, 2] = (f * inv_w) * (h * sinL + k * cosL)
    A[3, 2] = (s_2 * cosL) / (2 * w)
    A[4, 2] = (s_2 * sinL) / (2 * w)
    A[5, 2] = (h * sinL - k * cosL) * inv_w

    # Scale A by root_p_mu
    A *= root_p_mu

    # Compute vector b
    b = np.zeros(6)
    b[5] = np.sqrt(mu / p**3) * w**2

    # Return dynamics
    return A @ f_perturbation + b


@jit(nopython=True)
def j2_perturbation(elements):
    p = elements[0] # semi latus rectum [meters]
    f = elements[1]
    g = elements[2]
    h = elements[3]
    k = elements[4]
    L = elements[5] # true longitude [rad]
    # calculate useful values
    w = 1 + f*np.cos(L) + g*np.sin(L)
    r = p/w
    denominator = (1 + h**2 + k**2)**2
    pre_factor = -((MU_EARTH*EARTH_J2*EARTH_RADIUS_M**2)/(r**4))
    numerator_factor = h*np.sin(L) - k*np.cos(L)
    # calculate J2 acceleration in each RTN direction
    accel_r = pre_factor * (1.5) * (1 - ((12*numerator_factor**2)/denominator))
    accel_t = pre_factor * (12.) * ((numerator_factor*(h*np.cos(L)+k*np.sin(L)))/denominator)
    accel_n = pre_factor * (6.) * ((numerator_factor*(1 - h**2 - k**2))/denominator)
    return [accel_r, accel_t, accel_n]

def get_altitude(rv, radius_earth=6371):
    """
    Calculate altitude from state vector
    
    Parameters:
    rv : array-like
        State vector [x, y, z, vx, vy, vz]
    radius_earth : float, optional
        Radius of the Earth in km (default is 6371 km)
    
    Returns:
    float
        Altitude in km
    """
    r = rv[:3]
    return np.linalg.norm(r) - radius_earth

def get_velocity(rv):
    """
    Calculate velocity magnitude from state vector
    
    Parameters:
    rv : array-like
        State vector [x, y, z, vx, vy, vz]
    
    Returns:
    float
        Velocity magnitude in km/s
    """
    v = rv[3:]
    return np.linalg.norm(v)

def mee_to_lla(mee, mu=MU_EARTH):
    """
    Convert Modified Equinoctial Elements (MEE) to geodetic latitude, longitude, and altitude.
    
    Parameters:
    mee : array-like
        Modified Equinoctial Elements [p, f, g, h, k, L]
    mu : float, optional
        Gravitational parameter of the central body (default is Earth's gravitational parameter)
    
    Returns:
    float, float, float
        Geodetic latitude, longitude, and altitude in degrees, degrees, and kilometers, respectively
    """
    # Extract MEE elements
    p, f, g, h, k, L = mee
    
    # Compute semi-latus rectum
    a = p / (1 - f**2 - g**2)
    
    # Compute eccentricity vector
    e_vec = np.array([h, k, 0])
    
    # Compute eccentricity
    e = np.linalg.norm(e_vec)
    
    # Compute inclination
    i = 2 * np.arctan(np.sqrt(h**2 + k**2))
    
    # Compute right ascension of the ascending node
    Omega = np.arctan2(k, h)
    
    # Compute argument of latitude
    u = L - Omega
    
    # Compute true anomaly
    sin_nu = np.sqrt(h**2 + k**2) * np.sin(L - Omega) / (1 + f * np.cos(L) + g * np.sin(L))
    cos_nu = (f * np.cos(L) + g * np.sin(L) + 1) / (1 + f * np.cos(L) + g * np.sin(L))
    nu = np.arctan2(sin_nu, cos_nu)
    
    # Compute geodetic latitude
    lat = np.arctan2(np.sin(u) * np.sin(i), np.cos(u))
    
    # Compute longitude
    lon = Omega + nu
    
    # Compute altitude
    alt = a * (1 - e**2) / (1 + e * np.cos(nu)) - EARTH_RADIUS_KM
    
    # Convert latitude and longitude to degrees
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)
    
    return np.array([lat_deg, lon_deg, alt])


def lla_to_mee(lat, lon, alt, mu=MU_EARTH):
    """
    Convert geodetic latitude, longitude, and altitude to Modified Equinoctial Elements (MEE).
    
    Parameters:
    lat : float
        Geodetic latitude in degrees
    lon : float
        Longitude in degrees
    alt : float
        Altitude in kilometers
    mu : float, optional
        Gravitational parameter of the central body (default is Earth's gravitational parameter)
    
    Returns:
    np.ndarray
        Modified Equinoctial Elements [p, f, g, h, k, L]
    """
    # Convert latitude and longitude to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Compute Earth's radius at the given latitude
    r = EARTH_RADIUS_KM / np.sqrt(1 - EARTH_ECCENTRICITY**2 * np.sin(lat_rad)**2)
    
    # Compute position vector in ECEF frame
    x = (r + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (r + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (r * (1 - EARTH_ECCENTRICITY**2) + alt) * np.sin(lat_rad)
    
    # Compute velocity vector in ECEF frame
    v = np.sqrt(mu / r) / np.sqrt(1 + (2 * alt * r + alt**2) / mu)
    vx = -v * np.sin(lat_rad) * np.cos(lon_rad)
    vy = -v * np.sin(lat_rad) * np.sin(lon_rad)
    vz = v * np.cos(lat_rad)
    
    # Convert position and velocity vectors to Modified Equinoctial Elements (MEE)
    mee = rv_to_mee(np.array([x, y, z, vx, vy, vz]), mu)
    
    return mee

def rv_to_mee(rv, mu=MU_EARTH):
    """
    Convert state vector to Modified Equinoctial Elements (MEE).
    
    Parameters:
    rv : array-like
        State vector [x, y, z, vx, vy, vz]
    mu : float, optional
        Gravitational parameter of the central body (default is Earth's gravitational parameter)
    
    Returns:
    np.ndarray
        Modified Equinoctial Elements [p, f, g, h, k, L]
    """
    # Extract position and velocity vectors
    r = rv[:3]
    v = rv[3:]
    
    # Compute specific angular momentum
    h = np.cross(r, v)
    
    # Compute eccentricity vector
    e_vec = np.cross(v, h) / mu - r / np.linalg.norm(r)
    
    # Compute eccentricity
    e = np.linalg.norm(e_vec)
    
    # Compute inclination
    i = np.arccos(h[2] / np.linalg.norm(h))
    
    # Compute right ascension of the ascending node
    Omega = np.arctan2(h[0], -h[1])
    
    # Compute argument of latitude
    u = np.arctan2(r[2] / np.sin(i), r[0] * np.cos(Omega) + r[1] * np.sin(Omega))
    
    # Compute true anomaly
    nu = np.arctan2(np.dot(r, v) / np.linalg.norm(h), 1 - np.linalg.norm(r) / mu)
    
    # Compute semi-major axis
    a = 1 / (2 / np.linalg.norm(r) - np.linalg.norm(v)**2 / mu)
    
    # Compute MEE elements
    p = a * (1 - e**2)
    f = e_vec[0] * np.sin(u) - e_vec[1] * np.cos(u)
    g = e_vec[0] * np.cos(u) + e_vec[1] * np.sin(u)
    h = np.tan(i / 2) * np.sin(Omega)
    k = np.tan(i / 2) * np.cos(Omega)
    L = u + nu
    
    return np.array([p, f, g, h, k, L])