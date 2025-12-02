import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from simulator import run_simulation
import accel

# Physical constants in CGS units
AU = 1.496e13      # astronomical unit in cm
Msol = 1.989e33    # solar mass in grams
yr = 3.15576e7     # year in seconds
G = 6.6743e-8      # gravitational constant in cm^3 g^-1 s^-2

num_threads = accel.get_num_threads()
print(f"OpenMP threads available: {num_threads}")


N = int(1e3)
sphere_radius = float(10) * AU
total_mass = float(0.1) * Msol
n_years = float(10)
n_simulations = int(1)

# # Get simulation parameters from user
# N = int(input("Enter number of particles: "))
# sphere_radius = float(input("Enter sphere radius [AU]: ")) * AU
# total_mass = float(input("Enter total system mass [solar masses]: ")) * Msol
# n_years = float(input("Enter simulation time [years]: "))
# n_simulations = int(input("Enter number of simulations to run: "))

# Calculate derived parameters
particle_mass = total_mass / N  # each particle has equal mass
dt = 0.01 * yr  # timestep for leapfrog integration
n_step = int((n_years * yr) / dt)  # total number of timesteps

print(f"Particle mass: {particle_mass/Msol:.6e} solar masses")
print(f"Time step: {dt/yr} years")
print(f"Total steps: {n_step}")

def generate_sphere_particles(N, radius):
    '''
    Generate N particles uniformly distributed in a sphere.
    Uses rejection sampling: generate random points in cube, keep those inside sphere.
    
    Args:
        N: number of particles
        radius: sphere radius in cm
        
    Returns:
        positions: N x 3 array of particle positions in cm
        velocities: N x 3 array of particle velocities in cm/s (all zero)
    '''
    positions = np.zeros((N, 3))
    velocities = np.zeros((N, 3))  # all particles start at rest
    
    for i in range(N):
        # Rejection sampling to get uniform distribution in sphere
        while True:
            x = np.random.uniform(-radius, radius)
            y = np.random.uniform(-radius, radius)
            z = np.random.uniform(-radius, radius)
            r = np.sqrt(x**2 + y**2 + z**2)
            if r <= radius:  # accept if inside sphere
                positions[i] = [x, y, z]
                break
    
    return positions, velocities

# Run ensemble of simulations with different random initial conditions
all_dfs = []

for sim in range(n_simulations):
    print(f"\nSimulation {sim+1}/{n_simulations}")
    
    # Generate random initial particle distribution
    X0, V0 = generate_sphere_particles(N, sphere_radius)
    M = np.full(N, particle_mass)  # all particles have equal mass
    
    # Empty perturbation arrays (not used in N-body version)
    perturb_indices = np.array([], dtype=np.int32)
    perturb_pos = np.zeros((0, 3))
    perturb_vel = np.zeros((0, 3))
    
    # Run C++ simulation
    results = run_simulation(
        X0, V0, M,
        perturb_indices,
        perturb_pos,
        perturb_vel,
        sim + 1,  # simulation ID
        n_step,
        dt,
        yr
    )
    
    # Convert results array to DataFrame
    df_sim = pd.DataFrame(results, columns=[
        "simulation", "time_yr", "body_idx",
        "x_cm", "y_cm", "z_cm",
        "vx_cm_s", "vy_cm_s", "vz_cm_s",
        "KE", "PE"
    ])
    
    # Calculate total energy for each particle
    df_sim["E_tot"] = df_sim["KE"] + df_sim["PE"]
    all_dfs.append(df_sim)

# Combine all simulation results into single DataFrame
df = pd.concat(all_dfs, ignore_index=True)
df.to_csv("nbody_simulations.csv", index=False)
print("\nAll simulations complete. Data saved to nbody_simulations.csv")

# Create visualization plots for first simulation
# Shows behavior of first 10 particles
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

first_sim = df[df["simulation"] == 1]
plot_particles = min(10, N)  # plot up to 10 particles

for i in range(plot_particles):
    particle_df = first_sim[first_sim["body_idx"] == i]
    
    # Calculate radial distance from origin
    r = np.sqrt(particle_df["x_cm"]**2 + particle_df["y_cm"]**2 + particle_df["z_cm"]**2)
    axes[0, 0].plot(particle_df["time_yr"], r/AU, label=f"Particle {i}")
    
    # Calculate speed magnitude
    v = np.sqrt(particle_df["vx_cm_s"]**2 + particle_df["vy_cm_s"]**2 + particle_df["vz_cm_s"]**2)
    axes[0, 1].plot(particle_df["time_yr"], v/1e5, label=f"Particle {i}")
    
    # Plot XY trajectory
    axes[1, 0].plot(particle_df["x_cm"]/AU, particle_df["y_cm"]/AU, label=f"Particle {i}")
    
    # Plot total energy vs time
    axes[1, 1].plot(particle_df["time_yr"], particle_df["E_tot"], label=f"Particle {i}")

# Configure subplot 1: radial distance vs time
axes[0, 0].set_xlabel("Time [yr]")
axes[0, 0].set_ylabel("Radial Distance [AU]")
axes[0, 0].set_title("Radial Distance vs Time")
axes[0, 0].legend(fontsize='small')

# Configure subplot 2: speed vs time
axes[0, 1].set_xlabel("Time [yr]")
axes[0, 1].set_ylabel("Speed [km/s]")
axes[0, 1].set_title("Speed vs Time")
axes[0, 1].legend(fontsize='small')

# Configure subplot 3: XY trajectories
axes[1, 0].set_xlabel("X [AU]")
axes[1, 0].set_ylabel("Y [AU]")
axes[1, 0].set_title("XY Trajectories")
axes[1, 0].legend(fontsize='small')
axes[1, 0].axis('equal')  # equal aspect ratio

# Configure subplot 4: energy vs time
axes[1, 1].set_xlabel("Time [yr]")
axes[1, 1].set_ylabel("Total Energy [erg]")
axes[1, 1].set_title("Total Energy vs Time")
axes[1, 1].legend(fontsize='small')

plt.tight_layout()
plt.show()