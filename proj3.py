import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from simulator import run_simulation
from visualize_live import SimulationVisualizer
import accel
import os
import threading
from astropy.io import fits
from astropy.table import Table

# Physical constants in CGS units
AU = 1.496e13      # astronomical unit in cm
Msol = 1.989e33    # solar mass in grams
yr = 3.15576e7     # year in seconds
G = 6.6743e-8      # gravitational constant in cm^3 g^-1 s^-2

plot_dir = '/home/dyson/fall25/ASTRO142_Proj3/plots'
os.makedirs(plot_dir, exist_ok=True)

num_threads = accel.get_num_threads()
print(f"OpenMP threads available: {num_threads}")


# Visualization parameters
TRAIL_LENGTH = 50  # Number of historical positions to keep per particle
TARGET_FPS = 30    # Target frames per second for animation
WINDOW_WIDTH = 1400  # Total window width in pixels
WINDOW_HEIGHT = 800  # Total window height in pixels

# Hardcoded parameters, these seem to give a good chance of virial equilibrium 
N = int(10)
sphere_radius = float(0.001) * AU
total_mass = float(1e-16) * Msol
max_years = float(10000)
n_simulations = int(3)
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

print("=" * 60)
print("N-body simulation")
print("=" * 60)
print(f"Particles: {N}")
print(f"Total mass: {total_mass/Msol:.6e} solar masses")
print(f"Particle mass: {particle_mass/Msol:.6e} solar masses")
print(f"Initial sphere radius: {sphere_radius/AU:.6e} AU")
print(f"Collision radius: {collision_radius/AU:.6e} AU")
print(f"Timestep: {dt/yr:.4f} years")
print(f"Maximum steps: {max_step}")
print(f"Simulation duration: {max_years} years")
print(f"Number of simulations: {n_simulations}")
print("=" * 60)


def generate_sphere_particles(N, radius):
    """
    Generate N particles uniformly distributed in a sphere.
    Uses rejection sampling: generate random points in cube, keep those inside sphere.
    
    Args:
        N: number of particles
        radius: sphere radius in cm
        
    Returns:
        positions: N x 3 array of particle positions in cm
        velocities: N x 3 array of particle velocities in cm/s (all zero)
    """
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


def run_and_save_simulation(sim_id, fits_filename, append=False):
    """
    Run a single simulation and save to FITS file.
    
    Args:
        sim_id: simulation identifier (1-indexed)
        fits_filename: path to FITS file
        append: if True, append to existing file; if False, create new file
        
    Returns:
        DataFrame with simulation results
    """
    print(f"\nSimulation {sim_id}/{n_simulations}")
    
    # Generate initial conditions
    X0, V0 = generate_sphere_particles(N, sphere_radius)
    M = np.full(N, particle_mass)
    
    # Empty perturbation arrays (not used)
    perturb_indices = np.array([], dtype=np.int32)
    perturb_pos = np.zeros((0, 3))
    perturb_vel = np.zeros((0, 3))
    
    # Run simulation
    results = run_simulation(
        X0, V0, M,
        perturb_indices,
        perturb_pos,
        perturb_vel,
        sim_id,
        max_step,
        dt,
        yr,
        collision_radius
    )
    
    # Convert to DataFrame
    df_sim = pd.DataFrame(results, columns=[
        "simulation", "time_yr", "body_idx",
        "x_cm", "y_cm", "z_cm",
        "vx_cm_s", "vy_cm_s", "vz_cm_s",
        "KE", "PE"
    ])
    
    df_sim["E_tot"] = df_sim["KE"] + df_sim["PE"]
    
    # Convert to astropy Table for FITS
    table = Table.from_pandas(df_sim)
    
    if append:
        # Append to existing FITS file
        table_hdu = fits.BinTableHDU(table)
        table_hdu.header['SIMID'] = (sim_id, 'Simulation ID number')
        table_hdu.header['EXTNAME'] = f'SIM_{sim_id}'
        
        with fits.open(fits_filename, mode='append') as hdul:
            hdul.append(table_hdu)
            hdul.flush()
    else:
        # Create new FITS file with primary HDU
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['N_BODIES'] = N
        primary_hdu.header['SPHRAD'] = (sphere_radius, 'Initial sphere radius [cm]')
        primary_hdu.header['TOTMASS'] = (total_mass, 'Total system mass [g]')
        primary_hdu.header['DT'] = (dt, 'Timestep [s]')
        primary_hdu.header['MAXSTEP'] = max_step
        primary_hdu.header['COLLRAD'] = (collision_radius, 'Collision radius [cm]')
        primary_hdu.header['NSIMS'] = n_simulations
        
        # Create table HDU for first simulation
        table_hdu = fits.BinTableHDU(table)
        table_hdu.header['SIMID'] = (sim_id, 'Simulation ID number')
        table_hdu.header['EXTNAME'] = f'SIM_{sim_id}'
        
        hdul = fits.HDUList([primary_hdu, table_hdu])
        hdul.writeto(fits_filename, overwrite=True)
    
    print(f"Simulation {sim_id} complete and saved to FITS")
    
    return df_sim


def background_simulations(fits_filename, start_sim, end_sim):
    """
    Run simulations in background thread and append to FITS file.
    
    Args:
        fits_filename: path to FITS file
        start_sim: first simulation ID to run (inclusive)
        end_sim: last simulation ID to run (inclusive)
    """
    for sim_id in range(start_sim, end_sim + 1):
        run_and_save_simulation(sim_id, fits_filename, append=True)
    
    print("\nAll background simulations complete")


def create_static_plots(df, fits_filename):
    """
    Generate static analysis plots from all simulation data.
    
    Args:
        df: pandas DataFrame with all simulation data
        fits_filename: name of FITS file (for plot filenames)
    """
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
    plt.savefig(os.path.join(plot_dir, f"system_of_{N}_particles.png"), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    
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
    plt.savefig(os.path.join(plot_dir, f"Virial_for_{N}_particles.png"), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def main():
    """
    Main execution flow:
    1. Run simulation 1 and save to FITS
    2. Start background thread for simulations 2-N
    3. Visualize simulation 1 (blocking on main thread)
    4. Wait for background simulations to complete
    5. Generate static plots
    """
    fits_filename = 'nbody_simulations.fits'
    
    # Run first simulation and save to FITS
    print("\n" + "=" * 60)
    print("Running first simulation...")
    print("=" * 60)
    df_sim1 = run_and_save_simulation(1, fits_filename, append=False)
    
    # Start background thread for remaining simulations
    background_thread = None
    if n_simulations > 1:
        print("\n" + "=" * 60)
        print(f"Starting background thread for simulations 2-{n_simulations}...")
        print("=" * 60)
        background_thread = threading.Thread(
            target=background_simulations,
            args=(fits_filename, 2, n_simulations),
            daemon=False
        )
        background_thread.start()
    
    # Visualize first simulation on main thread
    print("\n" + "=" * 60)
    print("Starting visualization of simulation 1...")
    print("Close the visualization window to continue")
    print("=" * 60)
    
    visualizer = SimulationVisualizer(
        df_sim1, 
        sphere_radius,
        trail_length=TRAIL_LENGTH,
        target_fps=TARGET_FPS,
        window_width=WINDOW_WIDTH,
        window_height=WINDOW_HEIGHT
    )
    visualizer.run()
    
    print("\nVisualization window closed")
    
    # Wait for background simulations to complete
    if background_thread is not None:
        print("\nWaiting for background simulations to complete...")
        background_thread.join()
    
    # Read all data from FITS file for final plots
    print("\n" + "=" * 60)
    print("Reading all simulation data from FITS file...")
    print("=" * 60)
    
    all_dfs = []
    with fits.open(fits_filename) as hdul:
        for i in range(1, len(hdul)):
            table = hdul[i].data
            df = Table(table).to_pandas()
            all_dfs.append(df)
    
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\nAll simulations complete. Data saved to {fits_filename}")
    
    # Generate static plots
    print("\n" + "=" * 60)
    print("Generating static analysis plots...")
    print("=" * 60)
    create_static_plots(df_all, fits_filename)
    
    print("\nAll done!")


if __name__ == "__main__":
    main()