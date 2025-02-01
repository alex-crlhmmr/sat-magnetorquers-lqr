import numpy as np

def get_initial_state(orbital_elements):
    """
    Compute the 6D state vector of a satellite (position and velocity) in both
    the orbital plane (OP) and Earth-Centered Inertial (ECI) frames, given a set
    of classical orbital elements with mean anomaly at epoch.

    This function assumes a two-body Keplerian orbit and does not include
    perturbations (e.g., atmospheric drag, J2, third-body effects).

    Parameters
    ----------
    orbital_elements : dict
        Dictionary containing:
            - semi_major_axis (float) : Semi-major axis of the orbit [meters].
            - eccentricity    (float) : Eccentricity of the orbit [dimensionless].
            - inclination     (float) : Inclination i [degrees].
            - raan           (float) : Right Ascension of the Ascending Node, Ω [degrees].
            - arg_pe         (float) : Argument of Perigee, ω [degrees].
            - mean_anomaly   (float) : Mean Anomaly at epoch, M₀ [degrees].
            - mu             (float) : Gravitational parameter GM [m³/s²].
                                       For Earth, ~3.986004418e14.

    Returns
    -------
    state_dict : dict
        A dictionary with two keys, "OP" and "ECI", each containing
        a (6×1) NumPy array. The first three rows are position coordinates [m],
        and the last three rows are velocity coordinates [m/s].

        Example structure:
        {
          "OP":  [[x_orb], [y_orb], [z_orb], [vx_orb], [vy_orb], [vz_orb]],
          "ECI": [[x_eci], [y_eci], [z_eci], [vx_eci], [vy_eci], [vz_eci]]
        }

        - "OP"  : The 6D state vector in the orbital plane (x_orb, y_orb, z_orb, vx_orb, vy_orb, vz_orb).
                  Here, z_orb = 0 because we're in the 2D orbital plane.
        - "ECI" : The 6D state vector in an Earth-Centered Inertial frame using the 3-1-3
                  (z-x-z) rotation sequence: Rz(-Ω)·Rx(-i)·Rz(-ω).
    """ 
    
    # Extract orbital elements from the input dictionary
    a          = orbital_elements['semi_major_axis']   # [m]
    e          = orbital_elements['eccentricity']      # []
    i_deg      = orbital_elements['inclination']       # [deg]
    raan_deg   = orbital_elements['raan']              # [deg]
    arg_pe_deg = orbital_elements['arg_pe']            # [deg]
    M0_deg     = orbital_elements['mean_anomaly']      # [deg]
    mu         = orbital_elements['mu']                # [m^3/s^2]

    # Convert angles to radians
    i      = np.radians(i_deg)
    Omega  = np.radians(raan_deg)
    omega  = np.radians(arg_pe_deg)
    M0     = np.radians(M0_deg)

    # Solve Kepler's Equation (M = E - e sin E) for E given M0
    def solve_kepler_equation(M, ecc, tol=1e-10, max_iter=100):
        # Simple Newton-Raphson iteration
        # Initial guess:
        E = M if ecc < 0.8 else np.pi
        for _ in range(max_iter):
            f  = E - ecc * np.sin(E) - M
            df = 1 - ecc * np.cos(E)
            E_new = E - f / df
            if abs(E_new - E) < tol:
                return E_new
            E = E_new
        return E  # fallback if not converged

    E = solve_kepler_equation(M0, e)

    # True anomaly from eccentric anomaly
    nu = 2.0 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2.0),
        np.sqrt(1 - e) * np.cos(E / 2.0)
    )
    # Ensure the true anomaly is in the range [0, 2π)
    if nu < 0:
        nu += 2.0 * np.pi
    
    # Compute the orbital parameters
    p = a * (1 - e**2)
    # radial distance
    r = p / (1 + e * np.cos(nu))

    # Orbital-plane coordinates
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    z_orb = 0.0

    # Orbital-plane radial and tangential velocity components
    vr     = np.sqrt(mu / p) * e * np.sin(nu)
    vtheta = np.sqrt(mu / p) * (1 + e * np.cos(nu))

    # Convert (vr, vθ) into Cartesian coordinates in the orbital-plane
    vx_orb = vr     * np.cos(nu) - vtheta * np.sin(nu)
    vy_orb = vr     * np.sin(nu) + vtheta * np.cos(nu)
    vz_orb = 0.0

    # Rotation matrices for the orbital elements
    def rot_z(angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [ c, -s,  0],
            [ s,  c,  0],
            [ 0,  0,  1]
        ])

    def rot_x(angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [ 1,  0,  0],
            [ 0,  c, -s],
            [ 0,  s,  c]
        ])

    # Build the full rotation: Rz(-Ω) Rx(-i) Rz(-ω)
    R = rot_z(-Omega) @ rot_x(-i) @ rot_z(-omega)

    r_orb_vec = np.array([x_orb,  y_orb,  z_orb])
    v_orb_vec = np.array([vx_orb, vy_orb, vz_orb])

    # Rotate the orbital-plane coordinates into ECI
    r_eci_vec = R @ r_orb_vec
    v_eci_vec = R @ v_orb_vec

    # Each reference frame has its position and velocity vectors - the dict of dict associate ref frame - position/velocity to VECTORS not individual components
    state_dict = {
        "OP":  np.concatenate((r_orb_vec, v_orb_vec)).reshape(6,),
        "ECI": np.concatenate((r_eci_vec, v_eci_vec)).reshape(6,)
    }
    
    return state_dict



if __name__ == "__main__":

    # Example orbital parameters with satellite at perigee

    orbital_elements = {
        'semi_major_axis': 6378e3 + 400e3,  # [m]
        'eccentricity':    0.05,            # []
        'inclination':     80.0,            # [deg]
        'raan':            30.0,            # [deg]
        'arg_pe':          40.0,            # [deg] 
        'mean_anomaly':    0.0,             # [deg] - satelite at perigee
        'mu':              3.986004418e14   # [m^3/s^2]
    }

    initial_state = get_initial_state(orbital_elements)

    print("Orbital Plane initial state:")
    print(initial_state["OP"])
    print("ECI initial state:")
    print(initial_state["ECI"])



