import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from simulator import run_simulation
import accel
import os

# Physical constants in CGS units
AU = 1.496e13      # astronomical unit in cm
Msol = 1.989e33    # solar mass in grams
yr = 3.15576e7     # year in seconds
G = 6.6743e-8      # gravitational constant in cm^3 g^-1 s^-2

plot_dir = '/home/dyson/fall25/ASTRO142_Proj3/plots'
os.makedirs(plot_dir, exist_ok=True)

num_threads = accel.get_num_threads()
print(f"OpenMP threads available: {num_threads}")

N = int(4)
sphere_radius = float(0.001) * AU
total_mass = float(1e-15) * Msol
max_years = float(5000)
n_simulations = int(1)
collision_radius_factor = 0.01

# # Get simulation parameters from user
# N = int(input("Enter number of particles: "))
# sphere_radius = float(input("Enter sphere radius [AU]: ")) * AU
# total_mass = float(input("Enter total system mass [solar masses]: ")) * Msol
# n_years = float(input("Enter simulation time [years]: "))
# n_simulations = int(input("Enter number of simulations to run: "))

# Calculate derived parameters
particle_mass = total_mass / N
dt = 0.01 * yr
max_step = int((max_years * yr) / dt)
collision_radius = collision_radius_factor * sphere_radius

print(f"Particle mass: {particle_mass/Msol:.6e} solar masses")
print(f"Time step: {dt/yr} years")
print(f"Maximum steps: {max_step}")
print(f"Collision radius: {collision_radius/AU:.6e} AU")

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
    velocities = np.zeros((N, 3))
    
    for i in range(N):
        while True:
            x = np.random.uniform(-radius, radius)
            y = np.random.uniform(-radius, radius)
            z = np.random.uniform(-radius, radius)
            r = np.sqrt(x**2 + y**2 + z**2)
            if r <= radius:
                positions[i] = [x, y, z]
                break
    
    return positions, velocities

all_dfs = []

for sim in range(n_simulations):
    print(f"\nSimulation {sim+1}/{n_simulations}")
    
    X0, V0 = generate_sphere_particles(N, sphere_radius)
    M = np.full(N, particle_mass)
    
    perturb_indices = np.array([], dtype=np.int32)
    perturb_pos = np.zeros((0, 3))
    perturb_vel = np.zeros((0, 3))
    
    results = run_simulation(
        X0, V0, M,
        perturb_indices,
        perturb_pos,
        perturb_vel,
        sim + 1,
        max_step,
        dt,
        yr,
        collision_radius
    )
    
    df_sim = pd.DataFrame(results, columns=[
        "simulation", "time_yr", "body_idx",
        "x_cm", "y_cm", "z_cm",
        "vx_cm_s", "vy_cm_s", "vz_cm_s",
        "KE", "PE"
    ])
    
    df_sim["E_tot"] = df_sim["KE"] + df_sim["PE"]
    all_dfs.append(df_sim)

df = pd.concat(all_dfs, ignore_index=True)
df.to_csv("nbody_simulations.csv", index=False)
print("\nAll simulations complete. Data saved to nbody_simulations.csv")

first_sim = df[df["simulation"] == 1]
plot_particles = min(10, N)
plot_radius_factor = 3.0
plot_limit = plot_radius_factor * sphere_radius / AU

# First figure: particle-level plots (2x2)
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))

for i in range(plot_particles):
    particle_df = first_sim[first_sim["body_idx"] == i]
    
    r = np.sqrt(particle_df["x_cm"]**2 + particle_df["y_cm"]**2 + particle_df["z_cm"]**2)
    axes1[0, 0].plot(particle_df["time_yr"], r/AU, label=f"Particle {i}")
    
    v = np.sqrt(particle_df["vx_cm_s"]**2 + particle_df["vy_cm_s"]**2 + particle_df["vz_cm_s"]**2)
    axes1[0, 1].plot(particle_df["time_yr"], v/1e5, label=f"Particle {i}")
    
    scatter = axes1[1, 0].scatter(particle_df["x_cm"]/AU, particle_df["y_cm"]/AU, 
                                  c=particle_df["time_yr"], cmap='viridis', 
                                  s=2, alpha=0.6, label=f"Particle {i}")
    
    axes1[1, 1].plot(particle_df["time_yr"], particle_df["E_tot"], label=f"Particle {i}")

axes1[0, 0].axhline(y=sphere_radius/AU, color='k', linestyle='--', linewidth=1, label='Initial radius')
axes1[0, 0].set_xlabel("Time [yr]")
axes1[0, 0].set_ylabel("Radial Distance [AU]")
axes1[0, 0].set_title("Radial Distance vs Time")
axes1[0, 0].legend(fontsize='small')

axes1[0, 1].set_xlabel("Time [yr]")
axes1[0, 1].set_ylabel("Speed [km/s]")
axes1[0, 1].set_title("Speed vs Time")
axes1[0, 1].legend(fontsize='small')

circle = plt.Circle((0, 0), sphere_radius/AU, color='k', fill=False, linestyle='--', linewidth=1, label='Initial sphere')
axes1[1, 0].add_patch(circle)
axes1[1, 0].set_xlabel("X [AU]")
axes1[1, 0].set_ylabel("Y [AU]")
axes1[1, 0].set_title("XY Trajectories (colored by time)")
axes1[1, 0].axis('equal')
axes1[1, 0].set_xlim(-plot_limit, plot_limit)
axes1[1, 0].set_ylim(-plot_limit, plot_limit)
cbar = plt.colorbar(scatter, ax=axes1[1, 0], label='Time [yr]')
axes1[1, 0].legend(fontsize='small')

axes1[1, 1].set_xlabel("Time [yr]")
axes1[1, 1].set_ylabel("Total Energy [erg]")
axes1[1, 1].set_title("Total Energy vs Time")
axes1[1, 1].legend(fontsize='small')

plt.tight_layout()
plt.show()

# Second figure: system-level plots (1x2)
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

system_energy = first_sim.groupby('time_yr').agg({
    'KE': 'sum',
    'PE': 'sum',
    'E_tot': 'sum'
}).reset_index()

axes2[0].plot(system_energy['time_yr'], system_energy['KE'], label='Total KE', linewidth=1.5)
axes2[0].plot(system_energy['time_yr'], system_energy['PE'], label='Total PE', linewidth=1.5)
axes2[0].plot(system_energy['time_yr'], system_energy['E_tot'], label='Total Energy', linewidth=1.5, linestyle='--')
axes2[0].set_xlabel("Time [yr]")
axes2[0].set_ylabel("Energy [erg]")
axes2[0].set_title("System Energy vs Time")
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)

virial_ratio = np.abs(2.0 * system_energy['KE'] / system_energy['PE'])
axes2[1].plot(system_energy['time_yr'], virial_ratio, linewidth=1.5, color='purple')
axes2[1].axhline(y=1.0, color='k', linestyle='--', linewidth=1, label='Ideal virial (ratio=1)')
axes2[1].fill_between(system_energy['time_yr'], 0.95, 1.05, alpha=0.2, color='green', label='Equilibrium zone')
axes2[1].set_xlabel("Time [yr]")
axes2[1].set_ylabel("Virial Ratio |2*KE/PE|")
axes2[1].set_title("Virial Equilibrium Check")
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)
axes2[1].set_ylim(0, 2.5)

plt.tight_layout()
plt.show()