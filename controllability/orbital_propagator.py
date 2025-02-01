# orbital_propagator.py

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import TEME, ITRS, EarthLocation, SkyCoord

class OrbitalPropagator:
    def __init__(self, initial_state, orbit_params):
        """
        Initializes the Orbital Propagator with the initial state and orbital parameters.

        Parameters:
        initial_state (numpy.ndarray): Initial state vector [x, y, z, vx, vy, vz] in meters and m/s.
        orbit_params (dict): Dictionary containing orbital parameters.
        """
        self.state = initial_state.copy()
        self.orbit_params = orbit_params

    def gravitational_acceleration(self, r):
        """
        Calculates gravitational acceleration with J2 perturbation.

        Parameters:
        r (numpy.ndarray): Position vector in ECI frame (meters).

        Returns:
        numpy.ndarray: Acceleration vector in m/s^2.
        """
        mu = self.orbit_params['mu']        # Earth's gravitational parameter, m^3/s^2
        J2 = self.orbit_params['J2']        # Earth's J2 coefficient
        R_e = self.orbit_params['R_e']      # Earth's radius in meters

        norm_r = np.linalg.norm(r)
        x, y, z = r
        r2 = norm_r ** 2
        r3 = norm_r ** 3
        r5 = norm_r ** 5

        # J2 perturbation factor
        factor = (3/2) * J2 * (R_e / norm_r)**2

        ax = -mu * x / r3 * (1 - factor * (5 * z**2 / r2 - 1))
        ay = -mu * y / r3 * (1 - factor * (5 * z**2 / r2 - 1))
        az = -mu * z / r3 * (1 - factor * (5 * z**2 / r2 - 3))

        return np.array([ax, ay, az])

    def rk4_step(self, delta_t):
        """
        Performs a single RK4 integration step for orbital propagation.

        Parameters:
        delta_t (float): Time step in seconds.
        """
        def derivatives(state):
            r = state[:3]
            v = state[3:]
            a = self.gravitational_acceleration(r)
            return np.hstack((v, a))

        k1 = derivatives(self.state)
        k2 = derivatives(self.state + 0.5 * delta_t * k1)
        k3 = derivatives(self.state + 0.5 * delta_t * k2)
        k4 = derivatives(self.state + delta_t * k3)

        self.state += (delta_t / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def propagate_orbit(self, total_time, delta_t):
        """
        Propagates the orbit over the specified total time with given time steps.

        Parameters:
        total_time (float): Total propagation time in seconds.
        delta_t (float): Time step in seconds.

        Returns:
        tuple: (times, positions)
            - times (numpy.ndarray): Array of time stamps.
            - positions (numpy.ndarray): Array of position vectors over time.
        """
        num_steps = int(total_time / delta_t) + 1
        positions = np.zeros((num_steps, 3))
        times = np.zeros(num_steps)

        print(f"Starting propagation for {num_steps} steps with delta_t = {delta_t} seconds.")
        for step in range(num_steps):
            positions[step] = self.state[:3]
            times[step] = step * delta_t
            if step % 100 == 0 or step == num_steps - 1:
                print(f"Propagating step {step}/{num_steps - 1}")
            self.rk4_step(delta_t)
        print("Propagation loop completed.")

        return times, positions

def eci_to_geodetic(r_ECI, current_time_astropy):
    """
    Converts ECI coordinates to geodetic coordinates.

    Parameters:
    r_ECI (numpy.ndarray): Position vector in ECI frame (meters).
    current_time_astropy (astropy.time.Time): Current simulation time.

    Returns:
    tuple: (latitude in degrees, longitude in degrees, altitude in meters)
    """
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

def plot_3d_trajectory(positions):
    """
    Plots the 3D trajectory of the satellite.

    Parameters:
    positions (numpy.ndarray): Array of position vectors over time.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth for reference
    # Earth radius in meters
    R_e = 6378137.0
    # Create a sphere representing Earth
    u_sphere, v_sphere = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    x_sphere = R_e * np.cos(u_sphere) * np.sin(v_sphere)
    y_sphere = R_e * np.sin(u_sphere) * np.sin(v_sphere)
    z_sphere = R_e * np.cos(v_sphere)
    ax.plot_surface(x_sphere, y_sphere, z_sphere, cmap='Blues', alpha=0.6)

    # Plot satellite trajectory
    ax.plot(positions[:,0], positions[:,1], positions[:,2], color='red', label='Satellite Trajectory')

    # Set labels
    ax.set_xlabel('ECI X (m)')
    ax.set_ylabel('ECI Y (m)')
    ax.set_zlabel('ECI Z (m)')
    ax.set_title('Satellite 3D Trajectory Over Time')
    ax.legend()

    # Set equal aspect ratio
    max_range = np.array([positions[:,0].max()-positions[:,0].min(),
                          positions[:,1].max()-positions[:,1].min(),
                          positions[:,2].max()-positions[:,2].min()]).max() / 2.0

    mid_x = (positions[:,0].max()+positions[:,0].min()) * 0.5
    mid_y = (positions[:,1].max()+positions[:,1].min()) * 0.5
    mid_z = (positions[:,2].max()+positions[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


def main():
    # Define orbital parameters
    orbit_params = {
    'semi_major_axis': 6371e3 + 400e3,  # Earth's radius + 400 km, in meters
    'eccentricity': 0.1,                  # Elliptical orbit
    'inclination': 80.0,                  # Degrees (highly inclined)
    'raan': 0.0,                          # Right Ascension of Ascending Node (degrees)
    'arg_pe': 0.0,                        # Argument of Perigee (degrees)
    'true_anomaly_0': 0.0,                # True Anomaly at epoch (degrees)
    'mu': 3.986004418e14,                 # Earth's gravitational parameter, m^3/s^2
    'J2': 1.08263e-3,                     # Earth's J2 coefficient
    'R_e': 6378137.0                       # Earth's radius in meters
    }


    # Define initial simulation time (UTC)
    initial_time_str = '2024-01-01T00:00:00'
    initial_time_astropy = Time(initial_time_str, scale='utc')

    # Define initial state [x, y, z, vx, vy, vz] in meters and m/s
    # Calculate position and velocity at perigee for an elliptical orbit
    a = orbit_params['semi_major_axis']
    e = orbit_params['eccentricity']
    mu = orbit_params['mu']
    inclination_deg = orbit_params['inclination']
    inclination_rad = np.deg2rad(inclination_deg)

    # Position at perigee (x-direction)
    r_p = a * (1 - e)
    x_p = r_p
    y_p = 0.0
    z_p = 0.0

    # Velocity at perigee (y-direction) with inclination
    v_p = np.sqrt(mu * (1 + e) / (a * (1 - e)))
    vx_p = v_p * np.cos(inclination_rad)  # Inclined velocity component
    vy_p = v_p * np.sin(inclination_rad)
    vz_p = 0.0

    # Define initial state vector with correct inclination
    initial_state = np.array([
        x_p,    # x position in meters (r_p)
        y_p,    # y position in meters (0.0)
        z_p,    # z position in meters (0.0)
        0.0,    # vx velocity in m/s (0.0)
        v_p * np.cos(inclination_rad),  # vy velocity in m/s
        v_p * np.sin(inclination_rad)   # vz velocity in m/s
    ])



    # Initialize Orbital Propagator
    propagator = OrbitalPropagator(initial_state, orbit_params)

    # Define simulation parameters
    total_time = 86400.0   # Total propagation time in seconds (1 day)
    delta_t = 60.0         # Time step in seconds

    print("Starting orbital propagation...")
    # Propagate orbit
    times, positions = propagator.propagate_orbit(total_time, delta_t)
    print("Orbital propagation completed.")

    # Plot 3D trajectory
    print("Plotting 3D trajectory...")
    plot_3d_trajectory(positions)
    print("Plotting completed.")

    # Example: Get satellite position at a specific time after t0
    # For demonstration, let's get the position at 6 hours after t0
    target_time_seconds = 6 * 3600  # 6 hours in seconds
    step = int(target_time_seconds / delta_t)
    if step < len(positions):
        target_position = positions[step]
        print(f"Satellite position at {target_time_seconds} seconds after t0:")
        print(f"ECI X: {target_position[0]:.2f} m")
        print(f"ECI Y: {target_position[1]:.2f} m")
        print(f"ECI Z: {target_position[2]:.2f} m")
    else:
        print("Target time exceeds simulation duration.")

# define a function I can call from other file to propgate an orbit given orbital elements, initial date, time to progate, delata to propagate, initial state (baiscally main but packaged up)
# return postion and timestamp of those positions


if __name__ == "__main__":
    main()
