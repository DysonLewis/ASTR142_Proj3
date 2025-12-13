import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import os

AU = 1.496e13
yr = 3.15576e7

plot_dir = '/home/dyson/fall25/ASTRO142_Proj3/plots'
os.makedirs(plot_dir, exist_ok=True)


def extract_ensemble_statistics(fits_filename):
    """
    Extract final state and convergence statistics from all simulations in FITS file.
    
    Args:
        fits_filename: path to FITS file containing all simulation data
        
    Returns:
        DataFrame with one row per simulation containing summary statistics
    """
    stats_list = []
    
    with fits.open(fits_filename) as hdul:
        n_bodies = hdul[0].header['N_BODIES']
        initial_radius = hdul[0].header['SPHRAD']
        total_mass = hdul[0].header['TOTMASS']
        n_sims = hdul[0].header['NSIMS']
        
        print(f"Analyzing {n_sims} simulations with {n_bodies} particles each")
        
        for i in range(1, len(hdul)):
            if 'SIMID' not in hdul[i].header:
                continue
                
            sim_id = hdul[i].header['SIMID']
            print(f"Processing simulation {sim_id}...")
            
            sim_data = Table(hdul[i].data).to_pandas()
            
            times = sorted(sim_data['time_yr'].unique())
            final_time = times[-1]
            
            system_energy = sim_data.groupby('time_yr').agg({
                'KE': 'sum',
                'PE': 'sum'
            }).reset_index()
            
            system_energy['E_tot'] = system_energy['KE'] + system_energy['PE']
            system_energy['virial_ratio'] = np.abs(2.0 * system_energy['KE'] / system_energy['PE'])
            
            final_state = sim_data[sim_data['time_yr'] == final_time]
            final_KE = final_state['KE'].sum()
            final_PE = final_state['PE'].sum()
            final_E = final_KE + final_PE
            final_virial = np.abs(2.0 * final_KE / final_PE)
            
            initial_state = sim_data[sim_data['time_yr'] == times[0]]
            initial_E = initial_state['KE'].sum() + initial_state['PE'].sum()
            
            energy_drift = np.abs((final_E - initial_E) / initial_E) * 100
            
            peak_KE = system_energy['KE'].max()
            peak_PE_magnitude = np.abs(system_energy['PE']).max()
            
            in_equilibrium = (system_energy['virial_ratio'] >= 0.95) & (system_energy['virial_ratio'] <= 1.05)
            if in_equilibrium.any():
                first_equilibrium_idx = in_equilibrium.idxmax()
                time_to_virial = system_energy.loc[first_equilibrium_idx, 'time_yr']
            else:
                time_to_virial = np.nan
            
            final_positions = final_state[['x_cm', 'y_cm', 'z_cm']].values
            final_radii = np.sqrt(np.sum(final_positions**2, axis=1))
            max_radius = final_radii.max()
            mean_radius = final_radii.mean()
            
            final_velocities = final_state[['vx_cm_s', 'vy_cm_s', 'vz_cm_s']].values
            final_speeds = np.sqrt(np.sum(final_velocities**2, axis=1))
            max_speed = final_speeds.max()
            mean_speed = final_speeds.mean()
            
            stats_list.append({
                'sim_id': sim_id,
                'final_time_yr': final_time,
                'time_to_virial_yr': time_to_virial,
                'final_virial_ratio': final_virial,
                'final_KE': final_KE,
                'final_PE': final_PE,
                'final_E_tot': final_E,
                'energy_drift_percent': energy_drift,
                'peak_KE': peak_KE,
                'peak_PE_magnitude': peak_PE_magnitude,
                'max_radius_cm': max_radius,
                'mean_radius_cm': mean_radius,
                'max_speed_cm_s': max_speed,
                'mean_speed_cm_s': mean_speed
            })
    
    stats_df = pd.DataFrame(stats_list)
    return stats_df, n_bodies, initial_radius, total_mass


def create_ensemble_plots(stats_df, n_bodies, initial_radius, total_mass):
    """
    Create statistical summary plots from ensemble of simulations.
    
    Args:
        stats_df: DataFrame with statistics from each simulation
        n_bodies: number of particles per simulation
        initial_radius: initial sphere radius in cm
        total_mass: total system mass in g
    """
    n_sims = len(stats_df)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    valid_virial_times = stats_df['time_to_virial_yr'].dropna()
    if len(valid_virial_times) > 0:
        ax1.hist(valid_virial_times, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(valid_virial_times.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {valid_virial_times.mean():.1f} yr')
        ax1.axvline(valid_virial_times.median(), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {valid_virial_times.median():.1f} yr')
        ax1.set_xlabel('Time to Virial Equilibrium [yr]')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Equilibrium Times')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No simulations reached\nvirial equilibrium', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Distribution of Equilibrium Times')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(stats_df['final_virial_ratio'], bins=20, edgecolor='black', alpha=0.7, color='forestgreen')
    ax2.axvline(1.0, color='black', linestyle='-', linewidth=2, label='Ideal (1.0)')
    ax2.axvspan(0.95, 1.05, alpha=0.2, color='green', label='Equilibrium zone')
    ax2.axvline(stats_df['final_virial_ratio'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {stats_df["final_virial_ratio"].mean():.3f}')
    ax2.set_xlabel('Final Virial Ratio |2KE/PE|')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Final Virial Ratios')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(stats_df['energy_drift_percent'], bins=20, edgecolor='black', alpha=0.7, color='crimson')
    ax3.axvline(stats_df['energy_drift_percent'].mean(), color='darkred', linestyle='--', linewidth=2,
               label=f'Mean: {stats_df["energy_drift_percent"].mean():.4f}%')
    ax3.set_xlabel('Energy Drift [%]')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Total Energy Conservation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(stats_df['sim_id'], stats_df['final_KE'], s=50, alpha=0.6, color='blue', label='KE')
    ax4.scatter(stats_df['sim_id'], np.abs(stats_df['final_PE']), s=50, alpha=0.6, color='red', label='|PE|')
    ax4.scatter(stats_df['sim_id'], np.abs(stats_df['final_E_tot']), s=50, alpha=0.6, color='purple', label='|E_tot|')
    ax4.set_xlabel('Simulation ID')
    ax4.set_ylabel('Energy [erg]')
    ax4.set_title('Final Energies Across Simulations')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(stats_df['sim_id'], stats_df['peak_KE'], s=50, alpha=0.6, color='orange', label='Peak KE')
    ax5.scatter(stats_df['sim_id'], stats_df['peak_PE_magnitude'], s=50, alpha=0.6, color='teal', label='Peak |PE|')
    ax5.set_xlabel('Simulation ID')
    ax5.set_ylabel('Energy [erg]')
    ax5.set_title('Peak Energies Across Simulations')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(stats_df['sim_id'], stats_df['max_radius_cm']/AU, s=50, alpha=0.6, color='navy', label='Max radius')
    ax6.scatter(stats_df['sim_id'], stats_df['mean_radius_cm']/AU, s=50, alpha=0.6, color='skyblue', label='Mean radius')
    ax6.axhline(initial_radius/AU, color='gray', linestyle='--', linewidth=2, label='Initial radius')
    ax6.set_xlabel('Simulation ID')
    ax6.set_ylabel('Radius [AU]')
    ax6.set_title('Final Radial Extent')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.scatter(stats_df['sim_id'], stats_df['max_speed_cm_s']/1e5, s=50, alpha=0.6, color='darkgreen', label='Max speed')
    ax7.scatter(stats_df['sim_id'], stats_df['mean_speed_cm_s']/1e5, s=50, alpha=0.6, color='lightgreen', label='Mean speed')
    ax7.set_xlabel('Simulation ID')
    ax7.set_ylabel('Speed [km/s]')
    ax7.set_title('Final Particle Speeds')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.scatter(stats_df['final_virial_ratio'], stats_df['energy_drift_percent'], s=50, alpha=0.6, color='purple')
    ax8.axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax8.axvspan(0.95, 1.05, alpha=0.1, color='green')
    ax8.set_xlabel('Final Virial Ratio')
    ax8.set_ylabel('Energy Drift [%]')
    ax8.set_title('Energy Conservation vs Equilibrium')
    ax8.grid(True, alpha=0.3)
    
    ax9 = fig.add_subplot(gs[2, 2])
    summary_text = f"""
Ensemble summary
{'='*30}
Simulations: {n_sims}
Particles: {n_bodies}
Initial radius: {initial_radius/AU:.2e} AU
Total mass: {total_mass/1.989e33:.2e} M☉

Equilibrium
{'='*30}
Reached equilibrium: {(~stats_df['time_to_virial_yr'].isna()).sum()}/{n_sims}
Mean time: {stats_df['time_to_virial_yr'].mean():.1f} yr
Std time: {stats_df['time_to_virial_yr'].std():.1f} yr

Final state
{'='*30}
Mean virial ratio: {stats_df['final_virial_ratio'].mean():.3f}
Std virial ratio: {stats_df['final_virial_ratio'].std():.3f}

Mean energy drift: {stats_df['energy_drift_percent'].mean():.4f}%
Max energy drift: {stats_df['energy_drift_percent'].max():.4f}%
    """
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
            verticalalignment='top', horizontalalignment='left',
            fontfamily='monospace', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax9.axis('off')
    
    plt.suptitle(f'Ensemble Analysis: {n_sims} Simulations with {n_bodies} Particles', 
                fontsize=16, fontweight='bold')
    
    plt.savefig(os.path.join(plot_dir, f'ensemble_statistics_{n_sims}_sims.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nEnsemble plot saved to {plot_dir}")


def analyze_ensemble(fits_filename='nbody_simulations.fits'):
    """
    Main function to analyze ensemble and create plots.
    
    Args:
        fits_filename: path to FITS file with simulation results
    """
    print("Extracting ensemble statistics...")
    stats_df, n_bodies, initial_radius, total_mass = extract_ensemble_statistics(fits_filename)
    
    csv_filename = os.path.join(plot_dir, 'ensemble_statistics.csv')
    stats_df.to_csv(csv_filename, index=False)
    print(f"Statistics saved to {csv_filename}")
    
    print("\nCreating ensemble plots...")
    create_ensemble_plots(stats_df, n_bodies, initial_radius, total_mass)
    
    print("\nEnsemble analysis complete!")
    
    return stats_df