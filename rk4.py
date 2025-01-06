import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  
masses = np.array([
    1.989e30,   # Sun
    3.30e23,    # Mercury
    4.87e24,    # Venus
    5.972e24,   # Earth
    6.39e23,    # Mars
    1.898e27,   # Jupiter
    5.683e26,   # Saturn
    8.681e25,   # Uranus
    1.024e26,   # Neptune
    1.309e22    # Pluto
])

# Initial positions (m) and velocities (m/s)
positions = np.array([
    [0, 0, 0],                  
    [57.9e9, 0, 0],              
    [108.2e9, 0, 0],             
    [149.6e9, 0, 0],            
    [227.9e9, 0, 0],            
    [778.6e9, 0, 0],             
    [1.433e12, 0, 0],            
    [2.872e12, 0, 0],            
    [4.495e12, 0, 0],            
    [5.906e12, 0, 0]            
])

velocities = np.array([
    [0, 0, 0],                  
    [0, 47.4e3, 0],              
    [0, 35.0e3, 0],              
    [0, 29.8e3, 0],              
    [0, 24.1e3, 0],              
    [0, 13.1e3, 0],              
    [0, 9.7e3, 0],               
    [0, 6.8e3, 0],               
    [0, 5.4e3, 0],               
    [0, 4.7e3, 0]                
])

# Function to calculate acceleration
def acceleration(positions, masses):
    num_bodies = len(masses)
    acc = np.zeros_like(positions)
    for i in range(num_bodies):
        for j in range(num_bodies):
            if i != j:
                r_vec = positions[j] - positions[i]
                r_mag = np.linalg.norm(r_vec)
                acc[i] += G * masses[j] * r_vec / r_mag**3
    return acc

# RK4 Integration
def rk4_step(positions, velocities, masses, dt):
    acc1 = acceleration(positions, masses)
    k1_v = acc1
    k1_r = velocities

    acc2 = acceleration(positions + 0.5 * dt * k1_r, masses)
    k2_v = acc2
    k2_r = velocities + 0.5 * dt * k1_v

    acc3 = acceleration(positions + 0.5 * dt * k2_r, masses)
    k3_v = acc3
    k3_r = velocities + 0.5 * dt * k2_v

    acc4 = acceleration(positions + dt * k3_r, masses)
    k4_v = acc4
    k4_r = velocities + dt * k3_v

    new_positions = positions + dt * (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6
    new_velocities = velocities + dt * (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
    return new_positions, new_velocities

# adding parameters
dt = 3600 * 24  # Time step in seconds (1 day)
num_steps = 500  # Total number of steps (~500 days)
trajectories = np.zeros((num_steps, len(masses), 3))
trajectories[0] = positions

for step in range(1, num_steps):
    positions, velocities = rk4_step(positions, velocities, masses, dt)
    trajectories[step] = positions

def com(trajectories, masses):
    masses_reshaped = masses[np.newaxis, :, np.newaxis]
    com_positions = np.sum(trajectories * masses_reshaped, axis=1) / np.sum(masses)
    return trajectories - com_positions[:, np.newaxis, :]

trajectories_com = com(trajectories, masses)

# Separate inner and outer planets
inner_planets = [0, 1, 2, 3, 4]  # Sun, Mercury, Venus, Earth, Mars
outer_planets = [0, 5, 6, 7, 8, 9]  # Sun, Jupiter, Saturn, Uranus, Neptune, Pluto


fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Plot inner planets
for i in inner_planets:
    label = ["Sun", "Mercury", "Venus", "Earth", "Mars"][i]
    marker = 'x' if i == 0 else 'o'  # Cross for Sun, circles for planets
    axes[0].plot(trajectories_com[:, i, 0], trajectories_com[:, i, 1], label=label)
    axes[0].plot(trajectories_com[-1, i, 0], trajectories_com[-1, i, 1], marker=marker, markersize=8, alpha=0.8)  # Latest position

axes[0].set_title("Inner Planets")
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")
axes[0].legend()
axes[0].grid()
axes[0].axis('equal')

# Plot outer planets
for i in outer_planets:
    label = ["Sun", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"][outer_planets.index(i)]
    marker = 'x' if i == 0 else 'o' 
    axes[1].plot(trajectories_com[:, i, 0], trajectories_com[:, i, 1], label=label)
    axes[1].plot(trajectories_com[-1, i, 0], trajectories_com[-1, i, 1], marker=marker, markersize=8, alpha=0.8)  # Latest position

axes[1].set_title("Outer Planets")
axes[1].set_xlabel("x (m)")
axes[1].set_ylabel("y (m)")
axes[1].legend()
axes[1].grid()
axes[1].axis('equal')

plt.savefig('motion.pdf')
plt.tight_layout()
plt.show()
