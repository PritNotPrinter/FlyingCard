import pandas as pd
import matplotlib.pyplot as plt

# Load the simulation data from the CSV file
df = pd.read_csv("CardOptimizations.csv")

# Categorized based on user-specified throws
initialVelocity = df.iloc[2:5]   # Velocity 1, 2, 3
throwAngle = df.iloc[6:9]      # Angle 1, 2, 3
spinRate = pd.concat([df.iloc[0:2], df.iloc[[4]]])        # Spin 1, 2, True-Optimal

# Set up the plot grid
fig, axs = plt.subplots(3, 1, figsize=(8, 10))

# Distance vs Initial Velocity (Velocity Tests)
axs[0].scatter(initialVelocity["Initial Velocity (m/s)"], initialVelocity["Distance (m)"], marker='o', linestyle='-')
axs[0].set_title("Distance vs Initial Velocity")
axs[0].set_xlabel("Initial Velocity (m/s)", fontsize=9)
axs[0].set_ylabel("Distance (m)", fontsize=9)
axs[0].tick_params(labelsize=8)
axs[0].grid(True)

# Distance vs Throw Angle (Angle Tests)
axs[1].scatter(throwAngle["Throw Angle (deg)"], throwAngle["Distance (m)"], marker='o', linestyle='-', color='orange')
axs[1].set_title("Distance vs Throw Angle")
axs[1].set_xlabel("Throw Angle (deg)", fontsize=9)
axs[1].set_ylabel("Distance (m)", fontsize=9)
axs[1].tick_params(labelsize=8)
axs[1].grid(True)

# Distance vs Spin Rate (Spin Tests)
axs[2].scatter(spinRate["Spin Rate (rad/s)"], spinRate["Distance (m)"], marker='o', linestyle='-', color='green')
axs[2].set_title("Distance vs Spin Rate")
axs[2].set_xlabel("Spin Rate (rad/s)", fontsize=9)
axs[2].set_ylabel("Distance (m)", fontsize=9)
axs[2].tick_params(labelsize=8)
axs[2].grid(True)

plt.tight_layout(pad=2.5)
plt.show()
