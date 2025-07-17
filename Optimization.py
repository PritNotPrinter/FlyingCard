'''
Simulation for the Distance of Flying Playing Cards, for ALP Intro to Computational Python and Physics
Project Goal: Determine simulated distance of a playing card based on values of human-controllable variables

Specifications:
(1) Create simulated distance of initial variable values
(2) Create optimized variable values and display distance
(3) Determine variable disparity which results in the most distance reduction
(4) Take user input for specified distance and determine human-controlled variables for that distance
(5) Define restrictions based on the user input

This should all include physics concepts such as aerodynamics, drag, the magnus effect, gravity, spin, launch angle, etc. It should reflect the S.T. Yau Paper when possible for accuracy

Ranges:
- Modifiable Variables: m, th, initial velocity, throw angle, spin rate
- m (mass) = 0.00175 (in kgs)
- th (thickness) = 0.00027 (in meters)
- initial velocity <= 15 m/s
- throw angle >= 0 & throw angle <= 30 (degrees from perfectly horizontal)
- spin rate <= 50 (radians/second)

Minimize vs. Differential Evolution:
- Minimize is faster, differential is slower but more accurate
- Minimize searches locally -- the area around the predefined boundaries, while differential searches the entire boundary
- Since many of my results ignore the more extreme values when I don't want them to, switching to differential evolution can result in better accuracy for simulation and optimization results
'''


'''
Simulation for the Distance of Flying Playing Cards, for ALP Intro to Computational Python and Physics
Project Goal: Determine simulated distance of a playing card based on values of human-controllable variables

Specifications:
(1) Create simulated distance of initial variable values
(2) Create optimized variable values and display distance
(3) Determine variable disparity which results in the most distance reduction
(4) Take user input for specified distance and determine human-controlled variables for that distance
(5) Define restrictions based on the user input

This should all include physics concepts such as aerodynamics, drag, the magnus effect, gravity, spin, launch angle, etc. It should reflect the S.T. Yau Paper when possible for accuracy

Ranges:
- Modifiable Variables: m, th, initial velocity, throw angle, spin rate
- m (mass) = 0.00175 (in kgs)
- th (thickness) = 0.00027 (in meters)
- initial velocity <= 15 m/s
- throw angle >= 0 & throw angle <= 30 (degrees from perfectly horizontal)
- spin rate <= 50 (radians/second)

Minimize vs. Differential Evolution:
- Minimize is faster, differential is slower but more accurate
- Minimize searches locally -- the area around the predefined boundaries, while differential searches the entire boundary
- Since many of my results ignore the more extreme values when I don't want them to, switching to differential evolution can result in better accuracy for simulation and optimization results
'''


# =================== Imports ===================
import numpy as np  # Math operations
from scipy.integrate import solve_ivp  # ODE solver
from scipy.optimize import minimize  # Optimization algorithm
from scipy.optimize import differential_evolution # global optimization
import pandas as pd # exporting results as csv


# =================== Constants ===================
rho = 1.183             # Air density (kg/m^3)
BASE_Cd = 1.72          # Base drag coefficient (adjustable with spin)
A = 0.005544            # Surface area (m^2)
g = 9.81                # Gravity (m/s^2)
r = 0.0755              # Radius of card (m)
Ixy = 8.5407e-7         # Moment of inertia in xy-plane
Iz = 1.70814e-6         # Moment of inertia in z-axis
P = 0.1                 # Pressure difference (lift torque)


# =================== Motion Model ===================
def motion_model(t, Y, m, th, tau_0):
    x, y, z, x_dot, y_dot, z_dot, theta, theta_dot, phi, phi_dot, alpha, alpha_dot = Y

    angle_deg = np.degrees(theta)
    if angle_deg > 0:
        angle_penalty = 0.00001 * (angle_deg)**2
    else:
        angle_penalty = 1.0

    Cd = BASE_Cd * angle_penalty
    Cd *= (0.65 + 0.35 * (1 - np.exp(-alpha_dot / 25)))  # spin effect (moderate)

    Fx = -0.5 * rho * Cd * A * np.sin(theta) * x_dot**2

    if y_dot < 0:
        Fy_drag = 0.5 * rho * Cd * A * np.sin(phi) * y_dot**2
    else:
        Fy_drag = -0.5 * rho * Cd * A * np.sin(phi) * y_dot**2

    k_magnus = 0.02
    magnus_force = k_magnus * rho * x_dot * np.cos(theta) * (2 * np.pi * r**2) * alpha_dot * th * np.cos(phi)

    magnus_lift = 0.5 * rho * A * alpha_dot * x_dot
    Fy = Fy_drag - magnus_force + magnus_lift

    if z_dot < 0:
        Fz = -m * g + 0.5 * rho * Cd * 2 * A * np.cos(theta) * np.cos(phi) * z_dot**2
    else:
        Fz = -m * g - 0.5 * rho * Cd * 2 * A * np.cos(theta) * np.cos(phi) * z_dot**2

    tau_theta = P * A * r * np.sin(theta)
    theta_ddot = tau_theta / Ixy

    tau_phi = Iz * alpha_dot * phi_dot - P * A * r * np.sin(phi)
    phi_ddot = tau_phi / Ixy

    alpha_ddot = tau_0 / Iz

    return [
        x_dot, y_dot, z_dot,
        Fx / m, Fy / m, Fz / m,
        theta_dot, theta_ddot,
        phi_dot, phi_ddot,
        alpha_dot, alpha_ddot
    ]


# =================== Distance Simulation Function ===================

def simulate_distance(params, m, th, statsTracker=False):
    initialVelocity, throwAngle_deg, spinRate = params
    throwAngle = np.radians(throwAngle_deg)
    tau_0 = -1.7e-6

    Y0 = [
        0.0, 0.0, 1.8,
        initialVelocity * np.cos(throwAngle),
        0.17,
        initialVelocity * np.sin(throwAngle),
        throwAngle, 0.0,
        0.0, 0.0,
        0.0, spinRate
    ]

    def hit_ground(t, y):
        return y[2]
    hit_ground.terminal = True
    hit_ground.direction = -1

    sol = solve_ivp(
        lambda t, y: motion_model(t, y, m, th, tau_0),
        [0, 300], Y0,
        rtol=1e-5, atol=1e-5,
        events=hit_ground
    )

    x_vals = sol.y[0]
    distance = x_vals[-1] - x_vals[0]
    duration = sol.t[-1] - sol.t[0]

    if statsTracker:
        print(f"[LOG] θ = {throwAngle_deg:.2f}°, v = {initialVelocity:.2f} m/s, spin = {spinRate:.2f} rad/s\n"
              f"       ⇒ Distance: {distance:.2f} m | Duration: {duration:.2f} s")

    return -distance



# =================== Optimizer Runner ===================
def run_optimizations():
    cases = [
        (0.00175, 0.00027),  # single card
    ]

    results = []
    for i, (m, th) in enumerate(cases):
        print(f"\n--- Optimizing Setup {i+1} (m={m}, th={th}) ---")

        bounds = [(0.0, 15.0), (0.0, 30.0), (0.0, 50.0)]
        initial_guess = [(b[0] + b[1]) / 2 for b in bounds]

        # scipy.optimize.minimize algorithm for the optimization
        # result = minimize(
        #     simulate_distance,
        #     initial_guess,
        #     args=(m, th),
        #     bounds=bounds,
        #     method='L-BFGS-B'
        # )
      
        # differential evolution algorithm for the optimization
        result = differential_evolution(
            simulate_distance,
            bounds=bounds,        # Bounds are the main input for the search space
            args=(m, th),
            strategy='best1bin',  # A common, effective strategy
            maxiter=100,         # Maximum number of generations
            popsize=10,           # Number of individuals in the population
            tol=0.01              # Tolerance for convergence
        )

        final_dist = -result.fun
        print(f"Best Distance: {final_dist:.5f} m")
        print(f"Best Parameters: Initial Velocity = {result.x[0]:.5f}, Throw Angle = {result.x[1]:.5f}, Spin Rate = {result.x[2]:.5f}")

        # Append tuple: (distance, params)
        results.append((final_dist, result.x))

    return results




# =================== Simulated Distance for User Input ===================
def custom_throw(initialVelocity, throwAngle, spinRate, m, th):
    # Wrapper to simulate with known values
    dist = -simulate_distance([initialVelocity, throwAngle, spinRate], m, th)
    print(f"\nCustom simulation of a card with: \n"
          f"   Initial Velocity = {initialVelocity:.5f}\n"
          f"   Throw Angle = {throwAngle:.5f}\n"
          f"   Spin Rate = {spinRate:.5f}\n"
          f"Distance: {dist:.5f} m")
    store_simulation(initialVelocity, throwAngle, spinRate, dist)
    return dist


# =================== Find Most Influential Variable ===================
def optimize_variable(user_params, optimized_params, m, th):
    names = ["Initial Velocity", "Throw Angle", "Spin Rate"]  # Parameter names
    distances = []  # Track distances for single-variable swaps

    # Loop through each parameter and replace it with the optimized value
    for i in range(3):
        temp_params = list(user_params)
        temp_params[i] = optimized_params[i]
        dist = -simulate_distance(temp_params, m, th)
        distances.append(dist)

    # Identify variable with largest difference
    diffs = [abs(opt - user_params[i]) for i, opt in enumerate(optimized_params)]
    max_index = np.argmax(diffs)
    print(f"\nMost impactful variable: {names[max_index]} ({user_params[max_index]:.5f} as compared to {optimized_params[max_index]:.5f})\n\n\n")


# =================== Solve Input Variables for Distance ===================
def solve_for_distance(target_distance, results):
    m = 0.00175
    th = 0.00027

    # Grab the max possible distance from optimized single-card setup
    max_distance = results[0][0]
    max_params = results[0][1]

    # Check for impossible target distances
    if target_distance > max_distance:
        print("\nThis distance is not achievable with the existing parameter boundaries.")
        print(f"Maximum achievable distance: {max_distance:.5f} m")
        print(f"Achieved with:\n  Initial Velocity = {max_params[0]:.5f}, Throw Angle = {max_params[1]:.5f}, Spin Rate = {max_params[2]:.5f}")
        return None

    # Optimization objective: how close simulated distance is to the target
    def goal(params):
        return abs(-simulate_distance(params, m, th) - target_distance)

    bounds = [(0.0, 15.0), (0.0, 30.0), (0.0, 50.0)]
    initial_guess = [(b[0] + b[1]) / 2 for b in bounds]

    result = differential_evolution(goal, bounds=bounds, tol=1e-6, maxiter=100)
    final_dist = -simulate_distance(result.x, m, th)

    print()
    print(f"  Target Distance:   {target_distance:.5f} m")
    print(f"  Achieved Distance: {final_dist:.5f} m")
    print(f"  Params: Initial Velocity = {result.x[0]:.5f}, Throw Angle = {result.x[1]:.5f}, Spin Rate = {result.x[2]:.5f}")
    return result


# ================== Exporting to CSV ==================
aggregateSimulations = []
def store_simulation(velocity, angle, spin, distance):
    aggregateSimulations.append({
        "Initial Velocity (m/s)": round(velocity, 5),
        "Throw Angle (deg)": round(angle, 5),
        "Spin Rate (rad/s)": round(spin, 5),
        "Distance (m)": round(distance, 5)
    })
def export_csv():
    if aggregateSimulations:
        sorted_data = sorted(aggregateSimulations, key=lambda row: (row['Spin Rate (rad/s)'], row['Throw Angle (deg)'], row['Initial Velocity (m/s)']))
        df = pd.DataFrame(sorted_data)

        # Reorder columns
        column_order = ["Spin Rate (rad/s)", "Throw Angle (deg)", "Initial Velocity (m/s)", "Distance (m)"]
        df = df[column_order]

        df.to_csv("CardOptimizations.csv", index=False)
        print("Simulation logs exported to CardOptimizations.csv")
    else:
        print("Nothing to export.")

# =================== Run Everything ===================
if __name__ == "__main__":
    results = run_optimizations()

    # Test the default throw variables
    initialVelocity = 40.0
    throwAngle = 0.0
    spinRate = 50 
    user_params = [initialVelocity, throwAngle, spinRate]
    custom_throw(40.0, 0.0, 50, 0.00175, 0.00027) # Example -- should be ~65m distance
    custom_throw(0.0, 0.0, 50, 0.00175, 0.00027) # Velocity 1
    custom_throw(7.50, 0.0, 50, 0.00175, 0.00027) # Velocity 2
    custom_throw(15.0, 0.0, 50, 0.00175, 0.00027) # Velocity 3
    custom_throw(15.0, 0.0, 50, 0.00175, 0.00027) # Angle 1
    custom_throw(15.0, 15.0, 50, 0.00175, 0.00027) # Angle 2
    custom_throw(15.0, 30.0, 50, 0.00175, 0.00027) # Angle 3
    custom_throw(15.0, 0.0, 0, 0.00175, 0.00027) # Spin 1
    custom_throw(15.0, 0.0, 25, 0.00175, 0.00027) # Spin 2
    custom_throw(15.0, 0.0, 50, 0.00175, 0.00027) # True-Optimal values


    # Determine whichever variable reduces distance the most
    optimize_variable(user_params, results[0][1], 0.00175, 0.00027)
    # Aim for a target distance, in meters
    distanceInput = input("What is the target playing card distance to determine parameters for? \n")
    floatInput = float(distanceInput)
    solve_for_distance(floatInput, results)
    export_csv()
    print("\n\n\n\n")
