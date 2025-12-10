import numpy as np
import queue
import threading
import time
from simulator import run_simulation_live
from visualize_live import LiveSimulationVisualizer
import accel

# Physical constants in CGS units
AU = 1.496e13      # astronomical unit in cm
Msol = 1.989e33    # solar mass in grams
yr = 3.15576e7     # year in seconds

# Simulation parameters
N = 20                                   # Number of particles
sphere_radius = 0.001 * AU               # Initial sphere radius
total_mass = 1e-16 * Msol                # Total system mass
max_years = 10000                        # Simulation duration
collision_radius_factor = 0.01           # Collision radius as fraction of sphere radius
target_visualization_frames = 6000       # Target number of frames for visualization

# Calculate derived parameters
particle_mass = total_mass / N
dt = 0.01 * yr                           # Timestep
max_step = int((max_years * yr) / dt)    # Total integration steps
collision_radius = collision_radius_factor * sphere_radius

# Calculate visualization interval to get ~6000 frames
visualization_interval = max(1, max_step // target_visualization_frames)

print("=" * 60)
print("Rendering N-body simulation")
print("=" * 60)
print(f"Particles: {N}")
print(f"Total mass: {total_mass/Msol:.6e} solar masses")
print(f"Particle mass: {particle_mass/Msol:.6e} solar masses")
print(f"Initial sphere radius: {sphere_radius/AU:.6e} AU")
print(f"Collision radius: {collision_radius/AU:.6e} AU")
print(f"Timestep: {dt/yr:.4f} years")
print(f"Maximum steps: {max_step}")
print(f"Simulation duration: {max_years} years")
print(f"Visualization interval: every {visualization_interval} steps")
print(f"Expected visualization frames: ~{max_step // visualization_interval}")
print(f"OpenMP threads: {accel.get_num_threads()}")
print("=" * 60)


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


def simulation_thread(data_queue, speed_control):
    """
    Background thread that runs the C++ simulation and pushes frames to queue.
    
    Args:
        data_queue: thread-safe queue for passing frame data to visualization
        speed_control: dict with 'multiplier' key that controls simulation speed
    """
    print("\n[Simulation Thread] Starting simulation...")
    
    # Generate initial conditions
    X0, V0 = generate_sphere_particles(N, sphere_radius)
    M = np.full(N, particle_mass)
    
    # Empty perturbation arrays (not used in this simulation)
    perturb_indices = np.array([], dtype=np.int32)
    perturb_pos = np.zeros((0, 3))
    perturb_vel = np.zeros((0, 3))
    
    try:
        # Run simulation with live callback
        # The C++ code will call our callback function with frame data
        def frame_callback(time_yr, positions, velocities, KE, PE):
            """
            Callback function invoked by C++ simulator at each visualization frame.
            
            Args:
                time_yr: current simulation time in years
                positions: N x 3 array of positions in cm
                velocities: N x 3 array of velocities in cm/s
                KE: N-length array of kinetic energies in erg
                PE: N-length array of potential energies in erg
            """
            # Package data into tuple
            frame_data = (time_yr, positions.copy(), velocities.copy(), 
                         KE.copy(), PE.copy())
            
            # Push to queue (allows queue to grow as needed)
            data_queue.put(frame_data)
            
            # Apply speed control delay
            speed_mult = speed_control['multiplier']
            if speed_mult < 0.99:  # Not unlimited
                # Calculate delay: at 1x (mult=0), delay = 1/30 sec
                base_delay = 1.0 / 30.0
                delay = base_delay * (1.0 - speed_mult)
                time.sleep(delay)
        
        # Run simulation (this will be modified C++ function)
        run_simulation_live(
            X0, V0, M,
            perturb_indices,
            perturb_pos,
            perturb_vel,
            1,  # simulation ID
            max_step,
            dt,
            yr,
            collision_radius,
            visualization_interval,
            frame_callback
        )
        
        print("\n[Simulation Thread] Simulation complete!")
        
    except Exception as e:
        print(f"\n[Simulation Thread] Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main function that coordinates simulation and visualization threads.
    """
    # Create thread-safe queue for passing data
    data_queue = queue.Queue()
    
    # Shared speed control (dict so it's mutable across threads)
    speed_control = {'multiplier': 0.0}  # Start at 1x speed
    
    # Create and start simulation thread
    sim_thread = threading.Thread(
        target=simulation_thread,
        args=(data_queue, speed_control),
        daemon=True  # Thread will exit when main program exits
    )
    sim_thread.start()
    
    # Give simulation a moment to start
    time.sleep(0.5)
    
    # Create visualizer on main thread (matplotlib requires main thread)
    print("\n[Main Thread] Starting visualization...")
    visualizer = LiveSimulationVisualizer(N, sphere_radius, data_queue)
    
    # Connect speed control to visualizer's slider
    def update_speed_control(val):
        speed_control['multiplier'] = val
    
    visualizer.speed_slider.on_changed(update_speed_control)
    
    # Run visualization (blocking call)
    visualizer.run()
    
    # Wait for simulation thread to complete
    sim_thread.join(timeout=5.0)
    
    print("\n[Main Thread] Visualization closed. Exiting.")


if __name__ == "__main__":
    main()