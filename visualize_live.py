import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection
from collections import deque
import time

# Physical constants in CGS units
AU = 1.496e13      # astronomical unit in cm


class SimulationVisualizer:
    """
    Playback visualizer for pre-computed N-body simulation data.
    
    Displays particles in XY plane with trailing motion blur effect and
    a virial ratio indicator bar showing system energy balance.
    """
    
    def __init__(self, data_df, sphere_radius, visualization_interval=1, trail_length=50, 
                 target_fps=30, window_width=1400, window_height=800):
        """
        Initialize the visualization window and data structures.
        
        Args:
            data_df: pandas DataFrame with simulation data
            sphere_radius: initial sphere radius in cm
            visualization_interval: only display every Nth timestep from data
            trail_length: number of historical positions to keep per particle
            target_fps: target frames per second for animation
            window_width: total window width in pixels
            window_height: total window height in pixels
        """
        self.data_df = data_df
        self.sphere_radius = sphere_radius
        self.N = len(data_df['body_idx'].unique())
        self.visualization_interval = visualization_interval
        self.trail_length = trail_length
        self.target_fps = target_fps
        self.window_width = window_width
        self.window_height = window_height
        
        # Get unique time points and subsample by visualization_interval
        all_time_points = sorted(data_df['time_yr'].unique())
        self.time_points = all_time_points[::visualization_interval]
        self.n_frames = len(self.time_points)
        self.current_frame = 0
        
        # Trail data: each particle has deque of (x, y) positions
        self.particle_trails = [deque(maxlen=trail_length) for _ in range(self.N)]
        
        # Particle colors: use colormap for distinct colors
        self.colors = plt.cm.tab10(np.linspace(0, 1, self.N))
        
        # Playback control
        self.paused = False
        self.speed_multiplier = 0.0  # 0.0 = 1x speed, 1.0 = unlimited
        
        # FPS tracking
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Plot limits (auto-scale based on initial sphere)
        self.plot_limit = 2.0 * sphere_radius / AU
        
        self._setup_figure()
        
    def _setup_figure(self):
        """
        Create the matplotlib figure with two panels: XY plot and virial bar.
        """
        # Create figure with custom size
        self.fig = plt.figure(figsize=(self.window_width/100, self.window_height/100), dpi=100)
        
        # Create grid: main plot takes 85% width, virial bar takes 15%
        gs = self.fig.add_gridspec(2, 2, width_ratios=[85, 15], height_ratios=[9, 1],
                                   hspace=0.15, wspace=0.15)
        
        # Left panel: XY trajectory plot
        self.ax_xy = self.fig.add_subplot(gs[0, 0])
        self.ax_xy.set_xlabel("X [AU]", fontsize=12)
        self.ax_xy.set_ylabel("Y [AU]", fontsize=12)
        self.ax_xy.set_title("Particle Trajectories (XY Plane)", fontsize=14, fontweight='bold')
        self.ax_xy.set_xlim(-self.plot_limit, self.plot_limit)
        self.ax_xy.set_ylim(-self.plot_limit, self.plot_limit)
        self.ax_xy.set_aspect('equal')
        self.ax_xy.grid(True, alpha=0.3, linestyle='--')
        
        # Draw initial sphere boundary as reference
        initial_circle = plt.Circle((0, 0), self.sphere_radius/AU, 
                                   color='gray', fill=False, 
                                   linestyle='--', linewidth=2, 
                                   label='Initial sphere', alpha=0.5)
        self.ax_xy.add_patch(initial_circle)
        
        # Initialize line collections for trails (one per particle)
        self.trail_collections = []
        for i in range(self.N):
            lc = LineCollection([], linewidths=2, colors=self.colors[i])
            self.ax_xy.add_collection(lc)
            self.trail_collections.append(lc)
        
        # Current particle positions (scatter plot)
        self.particle_scatter = self.ax_xy.scatter([], [], s=30, c=[self.colors[0]], 
                                                   edgecolors='black', linewidths=1.5,
                                                   zorder=10, label='Particles')
        self.particle_scatter.set_offsets(np.empty((0, 2)))
        
        # Stats text overlay (top-left corner)
        self.stats_text = self.ax_xy.text(0.02, 0.98, '', 
                                          transform=self.ax_xy.transAxes,
                                          verticalalignment='top',
                                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                          fontsize=11, family='monospace')
        
        # Right panel: Virial ratio bar
        self.ax_virial = self.fig.add_subplot(gs[0, 1])
        self.ax_virial.set_xlim(0, 1)
        self.ax_virial.set_ylim(0.5, 1.5)
        self.ax_virial.set_ylabel("Virial Ratio |2KE/PE|", fontsize=12)
        self.ax_virial.set_title("Energy Balance", fontsize=12, fontweight='bold')
        self.ax_virial.set_xticks([])
        self.ax_virial.grid(True, axis='y', alpha=0.3)
        
        # Equilibrium zone (0.95 to 1.05) highlighted in green
        equilibrium_zone = patches.Rectangle((0, 0.95), 1, 0.1, 
                                            linewidth=0, edgecolor=None,
                                            facecolor='green', alpha=0.2,
                                            label='Equilibrium zone')
        self.ax_virial.add_patch(equilibrium_zone)
        
        # Center line at ratio = 1.0
        self.ax_virial.axhline(y=1.0, color='black', linestyle='-', linewidth=2, alpha=0.5)
        
        # Virial ratio indicator (sliding horizontal bar)
        self.virial_indicator = patches.Rectangle((0.1, 0.98), 0.8, 0.04,
                                                  linewidth=2, edgecolor='black',
                                                  facecolor='blue', alpha=0.8)
        self.ax_virial.add_patch(self.virial_indicator)
        
        # Virial ratio value text
        self.virial_text = self.ax_virial.text(0.5, 1.0, '1.00',
                                              ha='center', va='bottom',
                                              fontsize=14, fontweight='bold',
                                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Labels for top/bottom of bar
        self.ax_virial.text(0.5, 1.48, 'KE Dominated', ha='center', fontsize=9, style='italic')
        self.ax_virial.text(0.5, 0.52, 'PE Dominated', ha='center', fontsize=9, style='italic')
        
        # Speed control slider (bottom panel, spans both columns)
        self.ax_slider = self.fig.add_subplot(gs[1, :])
        self.ax_slider.set_title("Playback Speed Control", fontsize=10)
        
        # Create slider: 0.0 (1x speed) to 1.0 (unlimited speed)
        self.speed_slider = Slider(
            ax=self.ax_slider,
            label='Speed',
            valmin=0.0,
            valmax=1.0,
            valinit=0.0,
            valstep=0.05,
            color='skyblue'
        )
        
        # Custom slider labels
        self.ax_slider.text(0.0, -0.5, '1x', transform=self.ax_slider.transAxes, 
                           ha='left', va='top', fontsize=9)
        self.ax_slider.text(1.0, -0.5, 'Unlimited', transform=self.ax_slider.transAxes,
                           ha='right', va='top', fontsize=9)
        
        # Connect slider to callback
        self.speed_slider.on_changed(self._update_speed)
        
    def _update_speed(self, val):
        """
        Callback for speed slider changes.
        
        Args:
            val: slider value from 0.0 (1x) to 1.0 (unlimited)
        """
        self.speed_multiplier = val
        
    def _calculate_fps(self):
        """
        Calculate current frames per second based on recent frame times.
        
        Returns:
            fps: current FPS, or 0 if not enough data
        """
        current_time = time.time()
        self.frame_times.append(current_time)
        
        if len(self.frame_times) < 2:
            return 0.0
        
        # Calculate FPS from time span of stored frames
        time_span = self.frame_times[-1] - self.frame_times[0]
        if time_span > 0:
            fps = (len(self.frame_times) - 1) / time_span
            return fps
        return 0.0
    
    def _update_trails(self, positions):
        """
        Update particle trail positions and render with fading alpha.
        
        Args:
            positions: N x 2 array of current particle positions in AU
        """
        for i in range(self.N):
            # Add current position to this particle's trail
            x = positions[i, 0]
            y = positions[i, 1]
            self.particle_trails[i].append((x, y))
            
            # Convert deque to line segments for rendering
            if len(self.particle_trails[i]) > 1:
                trail_points = list(self.particle_trails[i])
                segments = [trail_points[j:j+2] for j in range(len(trail_points)-1)]
                
                # Create alpha gradient (older = more transparent)
                n_segments = len(segments)
                alphas = np.linspace(0.1, 1.0, n_segments)
                
                # Set colors with alpha gradient
                colors = np.zeros((n_segments, 4))
                colors[:, :3] = self.colors[i, :3]
                colors[:, 3] = alphas
                
                # Update line collection
                self.trail_collections[i].set_segments(segments)
                self.trail_collections[i].set_colors(colors)
    
    def _update_virial_bar(self, virial_ratio):
        """
        Update the virial ratio indicator bar position.
        
        Args:
            virial_ratio: current virial ratio value |2*KE/PE|
        """
        # Clamp to display range
        display_ratio = np.clip(virial_ratio, 0.5, 1.5)
        
        # Update indicator position (centered vertically at ratio value)
        self.virial_indicator.set_y(display_ratio - 0.02)
        
        # Color code: green near equilibrium, blue if PE dominated, red if KE dominated
        if 0.95 <= virial_ratio <= 1.05:
            color = 'green'
        elif virial_ratio < 0.95:
            color = 'blue'
        else:
            color = 'red'
        self.virial_indicator.set_facecolor(color)
        
        # Update text value
        self.virial_text.set_text(f'{virial_ratio:.2f}')
        self.virial_text.set_position((0.5, display_ratio))
    
    def update_frame(self, frame_num):
        """
        Animation update function called by FuncAnimation.
        
        Advances through pre-computed simulation data.
        
        Args:
            frame_num: frame number from FuncAnimation
        """
        # Check if we've reached the end
        if self.current_frame >= self.n_frames:
            return [self.particle_scatter, self.virial_indicator, 
                    self.virial_text, self.stats_text] + self.trail_collections
        
        # Get data for current time point
        current_time = self.time_points[self.current_frame]
        frame_data = self.data_df[self.data_df['time_yr'] == current_time]
        
        # Extract positions and energies
        positions_cm = frame_data[['x_cm', 'y_cm', 'z_cm']].values
        positions_au = positions_cm / AU
        KE = frame_data['KE'].values
        PE = frame_data['PE'].values
        
        # Update particle trails with fading effect
        self._update_trails(positions_au[:, :2])
        
        # Update current particle positions
        xy_positions = positions_au[:, :2]
        self.particle_scatter.set_offsets(xy_positions)
        
        # Calculate and update virial ratio
        total_KE = np.sum(KE)
        total_PE = np.sum(PE)
        if np.abs(total_PE) > 1e-30:
            virial_ratio = np.abs(2.0 * total_KE / total_PE)
        else:
            virial_ratio = 1.0
        self._update_virial_bar(virial_ratio)
        
        # Update stats text
        fps = self._calculate_fps()
        stats_str = f"Time: {current_time:.1f} yr\nFPS: {fps:.1f}\nFrame: {self.current_frame+1}/{self.n_frames}"
        self.stats_text.set_text(stats_str)
        
        # Apply speed control delay
        if self.speed_multiplier < 0.99:
            base_delay = 1.0 / self.target_fps
            delay = base_delay * (1.0 - self.speed_multiplier)
            time.sleep(delay)
        
        # Advance to next frame
        self.current_frame += 1
        
        return [self.particle_scatter, self.virial_indicator, 
                self.virial_text, self.stats_text] + self.trail_collections
    
    def run(self):
        """
        Start the animation loop.
        """
        from matplotlib.animation import FuncAnimation
        
        # Create animation with target FPS
        self.anim = FuncAnimation(self.fig, self.update_frame,
                                 frames=self.n_frames,
                                 interval=1000/self.target_fps,
                                 blit=True, 
                                 repeat=False,
                                 cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()