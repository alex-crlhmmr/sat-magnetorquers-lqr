from initial_state import get_initial_state
import numpy as np
import matplotlib.pyplot as plt


class OrbitalPropagator:
    def __init__(self, initial_state, propagator_params):
        """
        Initializes the Orbital Propagator with the initial state and orbital parameters.

        Parameters:
        initial_state (numpy.ndarray): Initial state vector [x, y, z, vx, vy, vz] in meters and m/s.
        propagator_params (dict): Dictionary containing propagator parameters.
        """
        self.state = initial_state.copy()
        self.propagator_params = propagator_params

    def gravitational_acceleration(self, r):
        """
        Calculates gravitational acceleration with J2 perturbation.

        Parameters:
        r (numpy.ndarray): Position vector in ECI frame (meters).

        Returns:
        numpy.ndarray: Acceleration vector in m/s^2.
        """
        mu = self.propagator_params['mu']        # Earth's gravitational parameter, m^3/s^2
        J2 = self.propagator_params['J2']        # Earth's J2 coefficient
        R_e = self.propagator_params['R_e']      # Earth's radius in meters

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

    def rk4_step(self):
        """
        Performs a single RK4 integration step for orbital propagation.

        """
        def derivatives(state):
            r = state[:3]
            v = state[3:]
            a = self.gravitational_acceleration(r)
            return np.hstack((v, a))

        k1 = derivatives(self.state)
        k2 = derivatives(self.state + 0.5 * self.propagator_params['delta_t'] * k1)
        k3 = derivatives(self.state + 0.5 * self.propagator_params['delta_t'] * k2)
        k4 = derivatives(self.state + self.propagator_params['delta_t'] * k3)

        self.state += (self.propagator_params['delta_t'] / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def propagate_orbit(self):
        """
        Propagates the orbit over the specified total time with given time steps.

        Returns:
        tuple: (times, positions)
            - times (numpy.ndarray): Array of time stamps.
            - states (numpy.ndarray): State vector at all propagation timesteps.
        """
        num_steps = int(self.propagator_params['total_time']  / self.propagator_params['delta_t'] ) + 1
        states = np.zeros((num_steps, 6))
        times = np.zeros(num_steps)

        print(f"Starting propagation for {num_steps} steps with delta_t = {self.propagator_params['delta_t']} seconds.")
        for step in range(num_steps):
            states[step] = self.state
            times[step] = step * self.propagator_params['delta_t']
            if step % 100 == 0 or step == num_steps - 1:
                print(f"Propagating step {step}/{num_steps - 1}")
            self.rk4_step()
        print("Propagation loop completed.")

        return times, states


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

if __name__ == "__main__":
    
    # Timestamp for initial state
    # t = Time("2021-03-01T12:00:00", scale='utc')
    
    # Example orbital parameters with satellite at perigee
    orbital_elements = {
        'semi_major_axis': 6378e3 + 400e3,  # [m]
        'eccentricity':    0.01,            # []
        'inclination':     30.0,            # [deg]
        'raan':            20.0,            # [deg]
        'arg_pe':          20.0,            # [deg] 
        'mean_anomaly':    0.0,             # [deg] - satelite at perigee
        'mu':              3.986004418e14   # [m^3/s^2]
    }
    initial_state = get_initial_state(orbital_elements)
    
    # Get initial state in ECI frame
    ECI_initial_state = initial_state['ECI']
    
    # Initialize the Orbital Propagator
    propagator_params = {
        'mu': 3.986004418e14,    # Earth's gravitational parameter, m^3/s^2
        'J2': 1.08263e-3,        # Earth's J2 coefficient
        'R_e': 6378137.0,         # Earth's radius in meters
        'total_time': 86400.0,   # Total propagation time in seconds (1 day)
        'delta_t': 60.0,         # Time step in seconds
    }
    propagator = OrbitalPropagator(ECI_initial_state, propagator_params)

    # Propagate the orbit
    print("Starting orbital propagation...")
    times, states = propagator.propagate_orbit()
    print("Orbital propagation completed.")

    # Plot 3D trajectory
    print("Plotting 3D trajectory...")
    plot_3d_trajectory(states[:, :3])
    print("Plotting completed.")

