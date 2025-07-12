'''
Simulation of Flying Playing Cards, for ALP Intro to Computational Python and Physics
Project Goal: Simulate 3D flight of playing card based on "Research on Motion of the Flying Playing Card" by S.T. Yau High School Science Award (Asia) research paper

Specifications:
- Simulate the rigid body translational and rotational motion of a playing card.
- Include effects of aerodynamic drag and gyroscopic stability.
- Don't incude holes in the cards (as the decks I throw), or sound emitted from flight.
- Incorporate differential equations for x(t), y(t), z(t), and Euler angles for pitch, yaw, and roll.
- Visualize somehow

Future Plans: Add more functionality based around:
- Holes in the cards, how that affects motion, airflow, sound emitted
- Including more throwing factors? e.g. best angle to throw at for distance, sound, wtv
- Some form of fluid dispersion/wake pattern around the card (Lattice Boltzmann?
'''




# =================== Imports ===================
import numpy as np  # Used for math operations and working with arrays
from scipy.integrate import solve_ivp  # Solves systems of differential equations
import matplotlib.pyplot as plt  # For making plots and graphs
import cmasher as cmr  # Makes the plot colors look better





# =================== Constants ===================
rho = 1.183  # How heavy air is (air density), in kg/m^3
Cd = 1.72  # Drag coefficient: how much the card resists air
A = 0.005544  # Area facing the air when flying, in square meters
m = 0.00175  # Mass of the playing card in kilograms
g = 9.81  # Gravity acceleration pulling the card down (m/s^2)
r = 0.0755  # Approximate radius of the card (used in spin calculations)
th = 0.00027  # Thickness of the card in meters
Ixy = 8.5407e-7  # Resistance to rotation (moment of inertia) around x/y axes
Iz = 1.70814e-6  # Resistance to rotation around z (spinning)
P = 0.1  # Pressure difference used for torque (simplified)
tau_0 = -1.7e-6  # Constant spin torque slowing down spin over time





# =================== ODE System ===================
def motion_model(t, Y):
    # Y holds all  values I'm tracking (position, velocity, rotation)
    x, y, z, x_dot, y_dot, z_dot, theta, theta_dot, phi, phi_dot, alpha, alpha_dot = Y

    # Force from air pushing against the card as it moves forward (x-axis)
    Fx = -0.5 * rho * Cd * A * np.sin(theta) * x_dot**2

    # Side force (y-axis), combining sideways drag and effect of spin
    if y_dot < 0:
        Fy = (0.5 * rho * Cd * A * np.sin(phi) * y_dot**2) - (
             rho * x_dot * np.cos(theta) * (2 * np.pi * r**2) * 15 * np.pi * th * np.cos(phi))
    else:
        Fy = (-0.5 * rho * Cd * A * np.sin(phi) * y_dot**2) - (
             rho * x_dot * np.cos(theta) * (2 * np.pi * r**2) * 15 * np.pi * th * np.cos(phi))
        
        
    # Add some sort of simplified Magnus effect
    # L_magnus = C_L * rho * v * r * alpha_dot (simplified )
    magnus_lift = 0.5 * rho * A * alpha_dot * x_dot  # relates to spin and forward speed
    Fy += magnus_lift

    # Up/down force (z-axis), including gravity and vertical air drag
    if z_dot < 0:
        Fz = -m * g + (0.5 * rho * Cd * 2 * A * np.cos(theta) * np.cos(phi) * z_dot**2)
    else:
        Fz = -m * g - (0.5 * rho * Cd * 2 * A * np.cos(theta) * np.cos(phi) * z_dot**2)

    # How the card tips forward/backward (pitch torque)
    tau_theta = P * A * r * np.sin(theta)
    theta_ddot = tau_theta / Ixy  # Newton's 2nd law for rotation

    # How the card turns left/right (yaw torque)
    tau_phi = Iz * alpha_dot * phi_dot - P * A * r * np.sin(phi)
    phi_ddot = tau_phi / Ixy

    # How the card spins like a frisbee (roll torque)
    alpha_ddot = tau_0 / Iz

    # This returns how each value is changing at that moment in time
    return [
        x_dot, y_dot, z_dot,          # Change in position = velocity
        Fx / m, Fy / m, Fz / m,       # Change in velocity = acceleration
        theta_dot, theta_ddot,        # Pitch angle and how fast it changes
        phi_dot, phi_ddot,            # Yaw angle and how fast it changes
        alpha_dot, alpha_ddot         # Roll angle and how fast it changes
    ]





# =================== Initial Conditions ===================
v0 = 7.0  # Speed at which card is thrown
Theta_0 = np.radians(10)  # Card starts angled 10 degrees upward
Omega_0 = 16 * np.pi  # Initial spinning speed (in radians/sec)

# This is where everything starts: position, speed, angles, and spins
Y0 = [
    0.0, 0.0, 1.25,                  # Starting location
    v0 * np.cos(Theta_0),            # Forward speed (x)
    0.17,                            # Slight sideways speed (y)
    v0 * np.sin(Theta_0),            # Upward speed (z)
    Theta_0, 0.0,                    # Pitch angle + how fast it's changing
    0.0, 0.0,                        # Yaw angle + how fast it's changing
    0.0, Omega_0                     # Roll angle + spin speed
]





# =================== Time Setup ===================
T_END = 5  # How long to simulate for
t_vals = np.linspace(0, T_END, 10000)  # Points in time to calculate motion

# Solve everything with the ivp library: calculate where/what the card does over time
sol = solve_ivp(
    motion_model, [0, T_END], Y0,
    t_eval=t_vals, rtol=1e-8, atol=1e-10
)





# =================== Visualization ===================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')  # 3D plot setup
ax.plot(sol.y[0], sol.y[1], sol.y[2], color="rebeccapurple", label='Simulated Trajectory')

# Variable line (testing different variable and constant values)
# Define a second set of throw conditions to compare
v0_2 = 7.0  # throw speed
Theta_0_2 = np.radians(10)  #  launch angle
Omega_0_2 = 8 * np.pi  #  spin
m_2 = 0.00175   # weight
A_2 = 0.005544   #  card area
r_2 = 0.0755   # radius
th_2 = 0.00027   # width
Cd_2 = 1.72  #  drag coefficient
P_2 = 0.1  # pressure differential
tau_0_2 = -1.7e-6  # Torque
Y0_2 = [
    0.0, 0.0, 1.25,
    v0_2 * np.cos(Theta_0_2),
    0.17,
    v0_2 * np.sin(Theta_0_2),
    Theta_0_2, 0.0,
    0.0, 0.0,
    0.0, Omega_0_2
]

# Temporarily replace global constants with new values
m_orig, A_orig, r_orig, th_orig, Cd_orig, P_orig, tau_0_orig = m, A, r, th, Cd, P, tau_0
m, A, r, th, Cd, P, tau_0 = m_2, A_2, r_2, th_2, Cd_2, P_2, tau_0_2

# Solve a second trajectory using new initial conditions
sol_2 = solve_ivp(
    motion_model, [0, T_END], Y0_2,
    t_eval=t_vals, rtol=1e-8, atol=1e-10
)

# Restore original constants
m, A, r, th, Cd, P, tau_0 = m_orig, A_orig, r_orig, th_orig, Cd_orig, P_orig, tau_0_orig

# Plot the second path with dashed line
ax.plot(sol_2.y[0], sol_2.y[1], sol_2.y[2], color='forestgreen', linestyle='--', label='Variable Path')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Flight Path of a Playing Card')
ax.legend()
plt.tight_layout()
plt.show()
