"""
Complementary Filter Parameter Sweep
Compare different alpha values to see their effect
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import butter, filtfilt


def lowpass_filter(data, cutoff=5, fs=100, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)


def load_data(filepath):
    df = pd.read_csv(filepath)

    print("\nApplying low-pass filter to magnetic field data...")
    df['Bx_filt'] = lowpass_filter(df['Bx'].values)
    df['By_filt'] = lowpass_filter(df['By'].values)
    df['Bz_filt'] = lowpass_filter(df['Bz'].values)

    df['Bx_cf'] = df['Bx_filt']
    df['By_cf'] = df['By_filt']
    df['Bz_cf'] = df['Bz_filt']

    df['B_mag'] = np.sqrt(df['Bx_cf']**2 + df['By_cf']**2 + df['Bz_cf']**2)

    # Compute qw from quaternion
    if all(col in df.columns for col in ['qx', 'qy', 'qz']):
        qx, qy, qz = df['qx'].values, df['qy'].values, df['qz'].values
        qw_squared = 1.0 - (qx**2 + qy**2 + qz**2)
        qw_squared = np.maximum(qw_squared, 0)
        df['qw'] = np.sqrt(qw_squared)

    return df


class EmpiricalSensorModel:
    """Empirical sensor model: v = a*b_mag^2 + b*b_mag + c"""

    def __init__(self, a=-0.0001523, b=0.03312, c=0.1211):
        self.a = a
        self.b = b
        self.c = c

    def b_mag_to_velocity(self, b_mag):
        """Convert magnetic field magnitude to velocity."""
        v = self.a * b_mag**2 + self.b * b_mag + self.c
        return np.maximum(v, 0)


class ComplementaryWindEstimator:
    """Complementary filter for wind estimation."""

    def __init__(self, alpha_velocity=0.7, alpha_wind=0.05, dt=0.02):
        self.alpha_v = alpha_velocity
        self.alpha_w = alpha_wind
        self.dt = dt

        # State
        self.velocity_world = np.zeros(3)
        self.wind_world = np.zeros(2)

    def update(self, velocity_odom, v_empirical, quat):
        """Update velocity and wind estimates."""
        # Convert body velocity to world frame
        try:
            rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            R_body_to_world = rot.as_matrix().T
            v_odom_world = R_body_to_world @ velocity_odom
        except:
            v_odom_world = velocity_odom

        # Get direction from odometry, magnitude from sensor
        v_odom_mag = np.linalg.norm(v_odom_world)

        if v_odom_mag > 0.01:
            v_direction = v_odom_world / v_odom_mag
            v_sensor_world = v_empirical * v_direction

            # Complementary filter for velocity
            self.velocity_world = (self.alpha_v * v_odom_world +
                                  (1 - self.alpha_v) * v_sensor_world)
        else:
            self.velocity_world = v_odom_world

        # Estimate wind from velocity difference
        v_fused_mag = np.linalg.norm(self.velocity_world)

        if v_fused_mag > 0.1:
            wind_indication = v_empirical - v_fused_mag

            if abs(wind_indication) < 1.0:
                wind_direction = self.velocity_world[:2] / (v_fused_mag + 1e-6)
                wind_increment = wind_indication * wind_direction

                self.wind_world = ((1 - self.alpha_w) * self.wind_world +
                                  self.alpha_w * wind_increment)
        else:
            self.wind_world *= 0.99

    def get_state(self):
        """Get current estimates."""
        return {
            'velocity_world': self.velocity_world.copy(),
            'wind_world': np.array([self.wind_world[0], self.wind_world[1], 0.0]),
            'wind_mag': np.linalg.norm(self.wind_world)
        }


def run_complementary_filter(data, empirical_model, alpha_velocity=0.7, alpha_wind=0.05):
    """Run complementary filter with given parameters."""

    # Time
    if 'time' in data.columns:
        time = data['time'].values
        dt = np.median(np.diff(time))
    else:
        dt = 0.02
        time = np.arange(len(data)) * dt

    if time[0] > 1000:
        time = time / 1000.0

    # Compute empirical velocity
    v_empirical = empirical_model.b_mag_to_velocity(data['B_mag'].values)

    # Initialize filter
    estimator = ComplementaryWindEstimator(
        alpha_velocity=alpha_velocity,
        alpha_wind=alpha_wind,
        dt=dt
    )

    # Storage
    v_mag_est = np.zeros(len(data))
    wind_mag = np.zeros(len(data))

    # Run filter
    for i in range(len(data)):
        # Get orientation
        if all(col in data.columns for col in ['Qx', 'Qy', 'Qz']):
            qw = np.sqrt(1 - data['Qx'].iloc[i] ** 2 + data['Qy'].iloc[i] ** 2 + data['Qz'].iloc[i] ** 2)
            quat = np.array([qw, data['Qx'].iloc[i],
                             data['Qy'].iloc[i], data['Qz'].iloc[i]])
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Get odometry velocity
        velocity_odom = np.array([data['Vx'].iloc[i],
                                 data['Vy'].iloc[i]])

        # Update
        estimator.update(velocity_odom, v_empirical[i], quat)

        # Store results
        state = estimator.get_state()
        v_mag_est[i] = np.linalg.norm(state['velocity_world'])
        wind_mag[i] = state['wind_mag']

    return time, v_mag_est, wind_mag, v_empirical


def plot_alpha_comparison(data, empirical_model, save_path='alpha_comparison.png'):
    """
    Run filter with different alpha values and plot comparison.
    """
    print("\n" + "="*70)
    print("ALPHA PARAMETER SWEEP")
    print("="*70)

    # Ground truth
    v_mag_true = np.sqrt(data['Vx']**2 + data['Vy']**2).values

    # Different alpha_velocity values to test
    alpha_values = [0.9, 0.7, 0.5, 0.3, 0.1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Keep wind alpha constant for fair comparison
    alpha_wind = 0.05

    # Run filter for each alpha
    results = {}
    print("\nRunning filters with different alpha_velocity values...")
    for alpha_v in alpha_values:
        print(f"  Testing alpha_velocity = {alpha_v}...")
        time, v_mag_est, wind_mag, v_empirical = run_complementary_filter(
            data, empirical_model,
            alpha_velocity=alpha_v,
            alpha_wind=alpha_wind
        )

        # Compute metrics
        rmse = np.sqrt(mean_squared_error(v_mag_true, v_mag_est))
        r2 = r2_score(v_mag_true, v_mag_est)

        results[alpha_v] = {
            'time': time,
            'v_mag_est': v_mag_est,
            'wind_mag': wind_mag,
            'rmse': rmse,
            'r2': r2
        }

        print(f"    RMSE: {rmse:.4f} m/s, R²: {r2:.4f}")

    # Store empirical for reference
    v_empirical_data = v_empirical

    # Create plots
    fig = plt.figure(figsize=(16, 14))

    # Create grid: 1 row for overview, then 3 rows x 2 cols for details
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3, height_ratios=[1.2, 1, 1, 1])

    # Adjust time
    time = results[alpha_values[0]]['time']
    if time[0] > 1000:
        time = time / 1000.0
    time = time - time[0]

    # Plot 1: All velocity estimates overlaid (top, spans both columns)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, v_mag_true, 'k-', linewidth=3, label='Ground Truth (CF EKF)', alpha=0.9, zorder=10)

    for i, alpha_v in enumerate(alpha_values):
        ax1.plot(time, results[alpha_v]['v_mag_est'],
                color=colors[i], linewidth=2,
                label=f'α={alpha_v}', alpha=0.8)

    ax1.plot(time, v_empirical_data, 'g--', linewidth=1.5,
            label='Flow Sensor (raw)', alpha=0.4)
    ax1.set_xlim([0, 25])
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Velocity Magnitude (m/s)', fontsize=12)
    ax1.set_title('Velocity Comparison: Different α_velocity Values\n(α_wind = 0.05 for all)',
                 fontweight='bold', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Plots 2-6: Individual comparisons (3 rows x 2 cols)
    for idx, alpha_v in enumerate(alpha_values):
        row = idx // 2 + 1  # Start from row 1 (0 is overview)
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])

        v_est = results[alpha_v]['v_mag_est']
        error = v_est - v_mag_true
        rmse = results[alpha_v]['rmse']
        r2 = results[alpha_v]['r2']

        # Plot estimate vs truth
        ax.plot(time, v_mag_true, 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
        ax.plot(time, v_est, color=colors[idx], linewidth=2,
               label=f'α_v={alpha_v}', alpha=0.8)
        ax.plot(time, v_empirical_data, 'g--', linewidth=1,
               label='Sensor', alpha=0.3)

        # Add error band
        ax.fill_between(time, v_mag_true - 0.1, v_mag_true + 0.1,
                       alpha=0.1, color='green', label='±0.1 m/s')

        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Velocity (m/s)', fontsize=10)
        ax.set_title(f'α_velocity = {alpha_v}\nRMSE: {rmse:.4f} m/s, R²: {r2:.4f}',
                    fontweight='bold', fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Complementary Filter: Alpha Parameter Comparison',
                fontsize=16, fontweight='bold', y=0.997)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {save_path}")
    plt.show()

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Alpha_v':<10} {'RMSE (m/s)':<15} {'R²':<10} {'Interpretation'}")
    print("-" * 70)

    interpretations = {
        0.9: "Very smooth, mostly odometry",
        0.7: "Balanced (recommended)",
        0.5: "Equal trust",
        0.3: "Trust sensor more",
        0.1: "Mostly sensor, noisy"
    }

    for alpha_v in alpha_values:
        rmse = results[alpha_v]['rmse']
        r2 = results[alpha_v]['r2']
        interp = interpretations.get(alpha_v, "")
        print(f"{alpha_v:<10.1f} {rmse:<15.4f} {r2:<10.4f} {interp}")

    # Also compute sensor-only RMSE
    rmse_sensor = np.sqrt(mean_squared_error(v_mag_true, v_empirical_data))
    r2_sensor = r2_score(v_mag_true, v_empirical_data)
    print(f"{'Sensor':<10} {rmse_sensor:<15.4f} {r2_sensor:<10.4f} {'Raw sensor (no fusion)'}")
    print(f"{'Odometry':<10} {0.0:<15.4f} {1.0:<10.4f} {'Ground truth reference'}")


def plot_wind_comparison(data, empirical_model, save_path='wind_alpha_comparison.png'):
    """
    Compare wind estimates with different alpha_wind values.
    """
    print("\n" + "="*70)
    print("WIND ALPHA PARAMETER SWEEP")
    print("="*70)

    # Different alpha_wind values to test
    alpha_wind_values = [0.01, 0.05, 0.1, 0.2]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Keep velocity alpha constant
    alpha_velocity = 0.7

    # Run filter for each alpha_wind
    results = {}
    print("\nRunning filters with different alpha_wind values...")
    for alpha_w in alpha_wind_values:
        print(f"  Testing alpha_wind = {alpha_w}...")
        time, v_mag_est, wind_mag, v_empirical = run_complementary_filter(
            data, empirical_model,
            alpha_velocity=alpha_velocity,
            alpha_wind=alpha_w
        )

        results[alpha_w] = {
            'time': time,
            'wind_mag': wind_mag,
            'wind_mean': np.mean(wind_mag),
            'wind_std': np.std(wind_mag)
        }

        print(f"    Mean wind: {results[alpha_w]['wind_mean']:.4f} ± {results[alpha_w]['wind_std']:.4f} m/s")

    # Adjust time
    time = results[alpha_wind_values[0]]['time']
    if time[0] > 1000:
        time = time / 1000.0
    time = time - time[0]

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, alpha_w in enumerate(alpha_wind_values):
        ax = axes[idx]
        wind = results[alpha_w]['wind_mag']
        mean_wind = results[alpha_w]['wind_mean']
        std_wind = results[alpha_w]['wind_std']

        ax.plot(time, wind, color=colors[idx], linewidth=2, alpha=0.8)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axhline(y=mean_wind, color='r', linestyle=':', linewidth=2,
                  label=f'Mean: {mean_wind:.4f} m/s')
        ax.fill_between(time, -0.2, 0.2, alpha=0.2, color='green',
                       label='Expected range (±0.2 m/s)')

        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Wind Magnitude (m/s)', fontsize=10)
        ax.set_title(f'α_wind = {alpha_w}\nMean: {mean_wind:.4f} ± {std_wind:.4f} m/s',
                    fontweight='bold', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.5, 1.0])


    plt.suptitle('Wind Estimation: Different α_wind Values\n(α_velocity = 0.7, indoors should be ~0)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    #plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nWind comparison plot saved: {save_path}")
    plt.show()

    # Summary
    print("\n" + "="*70)
    print("WIND ESTIMATION SUMMARY")
    print("="*70)
    print(f"{'Alpha_w':<10} {'Mean (m/s)':<15} {'Std (m/s)':<15} {'Interpretation'}")
    print("-" * 70)

    interpretations = {
        0.01: "Very smooth, slow adaptation",
        0.05: "Balanced (recommended)",
        0.1: "Responsive to changes",
        0.2: "Fast adaptation, noisy"
    }

    for alpha_w in alpha_wind_values:
        mean_w = results[alpha_w]['wind_mean']
        std_w = results[alpha_w]['wind_std']
        interp = interpretations.get(alpha_w, "")
        print(f"{alpha_w:<10.2f} {mean_w:<15.4f} {std_w:<15.4f} {interp}")


def main():
    filepath = r"C:\Users\ltjth\Documents\Research\UKF_Data\CF_ARR_Line_EKF0.3ms_FAN1.csv"

    print("="*70)
    print("COMPLEMENTARY FILTER - PARAMETER COMPARISON")
    print("="*70)

    empirical_model = EmpiricalSensorModel(a=-0.0001523, b=0.03312, c=0.1211)
    data = load_data(filepath)

    # Plot velocity comparison with different alphas
    plot_alpha_comparison(data, empirical_model, save_path='velocity_alpha_comparison.png')

    # Plot wind comparison with different alphas
    plot_wind_comparison(data, empirical_model, save_path='wind_alpha_comparison.png')

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("\nKey takeaways:")
    print("  - Higher α_velocity → smoother, closer to odometry")
    print("  - Lower α_velocity → tracks sensor more, noisier")
    print("  - Higher α_wind → faster wind adaptation, noisier")
    print("  - Lower α_wind → smoother wind estimate")
    print("\nRecommended: α_velocity = 0.7, α_wind = 0.05")


if __name__ == "__main__":
    main()
