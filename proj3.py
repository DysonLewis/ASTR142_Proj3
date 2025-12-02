import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use('TkAgg')  # Use Tk backend to avoid Qt warnings
import matplotlib.pyplot as plt
from astropy.coordinates import get_body_barycentric_posvel
from astropy import units as u
from astropy.time import Time
from accel import get_accel  # returns N x 3 accelerations in cm/s²
from simulator import run_simulation  # C++ simulation module

# Constants
AU = 1.496e13      # cm
Msol = 1.989e33
Mjup = 1.899e30
Mearth = 5.9742e27
yr = 3.15576e7     # seconds
G = 6.6743e-8      # cm^3 g^-1 s^-2

# Bodies
bodies = ["sun","mercury","venus","earth","mars","jupiter","saturn","uranus","neptune"]
N = len(bodies)

# Sublists, used later for plotting and such
rocky = bodies[1:5]
gas_giants = bodies[5:9]

# Initial positions/velocities
X0 = np.zeros((N,3))
V0 = np.zeros((N,3))
M = np.zeros(N)

apt = Time('2025-10-15 00:00')
for i, body in enumerate(bodies):
    pos, vel = get_body_barycentric_posvel(body, apt)
    X0[i,:] = pos.xyz.to(u.cm).value
    V0[i,:] = vel.xyz.to(u.cm/u.s).value

# Masses, no good way to loop this since astropy doesn't have masses?
M[0] = Msol
M[1] = 0.0553*Mearth
M[2] = 0.815*Mearth
M[3] = Mearth
M[4] = 0.107*Mearth
M[5] = Mjup
M[6] = 0.299*Mjup
M[7] = 0.046*Mjup
M[8] = 0.054*Mjup

# Random generator
seed = int(time.time()) # slightly better rng 
rng = np.random.default_rng(seed=seed)

# Perturbation preset
perturb_type = input(
    "Choose perturbation type ('small', 'large', 'custom'): "
).lower().strip()

if perturb_type == "small":
    perturb_planets = bodies[1:]
    perturb_pos_scale = 1e5
    perturb_vel_scale = 1e2
    n_orbits = 5  # force 5 orbits of Neptune, more takes too long since it's doing all planets "accurately"
    print(f"Small perturbation applied to all planets: pos={perturb_pos_scale} cm, vel={perturb_vel_scale} cm/s")
elif perturb_type == "large":
    perturb_planets = bodies[1:]
    perturb_pos_scale = 1e12 #e14 sometime works, sometimes flings planets
    perturb_vel_scale = 1e3
    n_orbits = 5  # force 5 orbits of Neptune
    print(f"Large perturbation applied to all planets: pos={perturb_pos_scale} cm, vel={perturb_vel_scale} cm/s")
elif perturb_type == "custom":
    perturb_input = input(
        "Enter planets to perturb (comma-separated, e.g., earth,jupiter) or 'all': "
    ).lower().strip()
    if perturb_input == "all":
        perturb_planets = bodies[1:]
    else:
        perturb_planets = [p.strip() for p in perturb_input.split(",")]

    perturb_pos_scale = float(input("Enter position perturbation scale [cm]: "))
    perturb_vel_scale = float(input("Enter velocity perturbation scale [cm/s]: "))
    # for custom, ask for orbits later based on outermost
else:
    raise ValueError("Invalid perturbation type. Please enter 'small', 'large', or 'custom'.")

# Determine timestep dt
if set(perturb_planets).issubset(rocky) and "mercury" not in perturb_planets:
    dt = 0.01 * yr
elif perturb_planets == ["mercury"]:
    # needed to keep mercury from getting launched out, still happens though
    dt = 0.001 * yr
elif set(perturb_planets).issubset(gas_giants):
    dt = 0.1 * yr # periods are longer so don't need small time steps
else:
    dt = 0.01 * yr # this makes simulations take a long time

print(f"Using dt = {dt/yr} yr")

# Determine outermost perturbed planet
if perturb_type == "custom" and perturb_planets:
    outermost_planet = max(perturb_planets, key=lambda p: bodies.index(p))
    n_orbits = int(input(f"Enter number of orbits for {outermost_planet.capitalize()}: "))
elif perturb_type in ["small", "large"]:
    outermost_planet = "neptune"
else:
    outermost_planet = None
    n_orbits = 1

# Index map
planet_idx = {body: i for i, body in enumerate(bodies)}

# Determine simulation years based on outermost planet
outer_idx = planet_idx[outermost_planet]
r0 = np.linalg.norm(X0[outer_idx] - X0[0])
T_sec = 2 * np.pi * np.sqrt(r0**3 / (G * M[0]))
T_yr = T_sec / yr
n_years = n_orbits * T_yr
print(f"Simulating {n_years:.2f} years (~{n_orbits} orbits of {outermost_planet.capitalize()})")

n_step = int((n_years * yr) / dt)

# Prepare perturbation arrays for C++
perturb_indices = []
for planet in perturb_planets:
    if planet in planet_idx:
        perturb_indices.append(planet_idx[planet])
    else:
        print(f"Warning: '{planet}' not recognized. Skipping.")

perturb_indices = np.array(perturb_indices, dtype=np.int32)
n_perturb = len(perturb_indices)

# Number of simulations
n_simulations = int(input("Enter number of simulations to run: "))

# Run simulations
all_dfs = []

for sim in range(n_simulations):
    print(f"\n=== Simulation {sim+1}/{n_simulations} ===")
    
    # Generate perturbations
    perturb_pos = rng.normal(scale=perturb_pos_scale, size=(n_perturb, 3))
    perturb_vel = rng.normal(scale=perturb_vel_scale, size=(n_perturb, 3))
    
    # Run simulation in C++
    results = run_simulation(
        X0, V0, M,
        perturb_indices,
        perturb_pos,
        perturb_vel,
        sim + 1,
        n_step,
        dt,
        yr
    )
    
    # Convert results to DataFrame
    # Results shape: (n_step * N, 11)
    # Columns: sim_id, time_yr, body_idx, x, y, z, vx, vy, vz, KE, PE
    df_sim = pd.DataFrame(results, columns=[
        "simulation", "time_yr", "body_idx",
        "x_cm", "y_cm", "z_cm",
        "vx_cm_s", "vy_cm_s", "vz_cm_s",
        "KE", "PE"
    ])
    
    # Add body names and total energy
    df_sim["body"] = df_sim["body_idx"].apply(lambda idx: bodies[int(idx)])
    df_sim["E_tot"] = df_sim["KE"] + df_sim["PE"]
    df_sim = df_sim.drop(columns=["body_idx"])
    
    all_dfs.append(df_sim)

# Combine all simulations
df = pd.concat(all_dfs, ignore_index=True)
df.to_csv("solar_system_simulations.csv", index=False)
print("\nAll simulations complete. Data saved to solar_system_simulations.csv")

# putting two plots together
fig, axes = plt.subplots(1, 2, figsize=(18, 9))  # 1 row, 2 columns

def plot_orbits(ax, bodies_to_plot, df, n_simulations, title):
    # Plot orbits for the given bodies on the provided Axes object.
    for body in bodies_to_plot:
        if body in planet_idx:  # skip invalid names
            for sim in range(n_simulations):
                planet_df = df[(df["body"] == body) & (df["simulation"] == sim+1)]
                ax.plot(
                    planet_df["x_cm"]/AU,
                    planet_df["y_cm"]/AU,
                    label=f"{body.capitalize()} (Sim {sim+1})"
                )
    ax.set_xlabel("X [AU]")
    ax.set_ylabel("Y [AU]")
    ax.set_title(
        f"{title}\nPerturbation: pos={perturb_pos_scale:.2e} cm, vel={perturb_vel_scale:.2e} cm/s, "
        f"Time ≈ {n_years:.2f} yr"
    )
    ax.legend(fontsize='small')
    ax.axis("equal")
    

# Left: Gas giants
plot_orbits(axes[0], gas_giants, df, n_simulations, "Gas Giant Orbits")

# Right: Perturbed planets if custom, rocky planets if small or large preset
if perturb_type == 'custom' and perturb_input != 'all':
    plot_orbits(axes[1], perturb_planets, df, n_simulations, "Perturbed Planets")
elif perturb_type in ('small', 'large') or perturb_input == 'all':
    plot_orbits(axes[1], rocky, df, n_simulations, "Rocky Planets")

plt.tight_layout()
plt.show()