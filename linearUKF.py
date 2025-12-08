# """
# Simple UKF for Velocity estimation using magnetic whisker sensor
# No wind estimation, no theta-based model
# Direct prediction of Velocity from magnetic field measurements
# Includes angular velocity (omega) in measurement model
# """
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from sklearn.metrics import mean_squared_error, r2_score


def load_data(filepath):
    """
    Load data from csv.

    Expected columns:
    - time or timestamp
    - Bx, By, Bz: Magnetic field measurements
    - Vx, Vy, Vz: Ground truth Velocity (body frame)
    - omega_x, omega_y, omega_z: Angular Velocity
    - qx, qy, qz: Quaternion components (qw will be computed)
    """
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    print(f"Loaded {len(df)} data points")
    print(f"Columns: {list(df.columns)}")

    # Check required columns
    required = ['Bx', 'By', 'Bz']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Apply Crazyflie coordinate transformation
    print("\nApplying Crazyflie coordinate transformation...")
    df['Bx_cf'] = -df['Bx']
    df['By_cf'] = -df['By']
    df['Bz_cf'] = df['Bz']

    # Compute qw from qx, qy, qz (quaternion normalization)
    if all(col in df.columns for col in ['qx', 'qy', 'qz']):
        print("Computing qw from quaternion components...")
        qx, qy, qz = df['qx'].values, df['qy'].values, df['qz'].values
        # qw^2 = 1 - qx^2 - qy^2 - qz^2
        qw_squared = 1.0 - (qx**2 + qy**2 + qz**2)
        qw_squared = np.maximum(qw_squared, 0)  # Avoid negative due to numerical errors
        df['qw'] = np.sqrt(qw_squared)
        print(f"  qw range: [{df['qw'].min():.4f}, {df['qw'].max():.4f}]")

    print(f"\nMagnetic field statistics:")
    print(f"  Bx: [{df['Bx'].min():.2f}, {df['Bx'].max():.2f}]")
    print(f"  By: [{df['By'].min():.2f}, {df['By'].max():.2f}]")
    print(f"  Bz: [{df['Bz'].min():.2f}, {df['Bz'].max():.2f}]")

    if 'Vx' in df.columns:
        V_mag = np.sqrt(df['Vx']**2 + df['Vy']**2 + df['Vz']**2)
        print(f"\nVelocity magnitude: [{V_mag.min():.3f}, {V_mag.max():.3f}] m/s")

    # Check for omega
    if all(col in df.columns for col in ['omega_x', 'omega_y', 'omega_z']):
        omega_mag = np.sqrt(df['omega_x']**2 + df['omega_y']**2 + df['omega_z']**2)
        print(f"Angular velocity magnitude: [{omega_mag.min():.3f}, {omega_mag.max():.3f}] rad/s")

    return df


class VelocityUKF:
    """
    Simple UKF for Velocity estimation from magnetic sensor.

    State: [Vx, Vy, Vz] - Velocity in body frame
    Measurement: [Bx, By, Bz] - magnetic field (transformed for Crazyflie)
    """

    def __init__(self, dt=0.02, process_noise=0.1, measurement_noise=1.0):
        """
        Initialize UKF.

        Parameters:
        -----------
        dt : float
            Time step (seconds)
        process_noise : float
            Process noise for Velocity states
        measurement_noise : float
            Measurement noise for magnetic field
        """
        self.dt = dt
        self.dim_x = 3  # State: [Vx, Vy, Vz]
        self.dim_z = 3  # Measurement: [Bx, By, Bz]

        # Create sigma points
        points = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.1, beta=2.0, kappa=0.0)

        # Initialize UKF
        self.ukf = UKF(dim_x=self.dim_x, dim_z=self.dim_z, dt=dt,
                       fx=self.state_transition,
                       hx=self.measurement_function,
                       points=points)

        # Initial state: [Vx, Vy, Vz]
        self.ukf.x = np.zeros(3)

        # Initial covariance
        self.ukf.P = np.eye(3) * 1.0

        # Process noise (Velocity changes)
        self.ukf.Q = np.eye(3) * (process_noise ** 2)

        # Measurement noise (magnetic field sensor noise)
        self.ukf.R = np.eye(3) * (measurement_noise ** 2)

        # Learned mapping parameters
        # b = A_v * v + A_omega * omega + b0
        self.A_v = np.eye(3)  # Velocity mapping matrix
        self.A_omega = np.zeros((3, 3))  # Angular velocity mapping matrix
        self.b0 = np.zeros(3)  # Offset

        # Store current omega for measurement function
        self.current_omega = np.zeros(3)

    def set_mapping_params(self, A_v, A_omega, b0):
        """
        Set the learned mapping from velocity and omega to magnetic field.

        Parameters:
        -----------
        A_v : array (3, 3)
            Velocity mapping matrix
        A_omega : array (3, 3)
            Angular velocity mapping matrix
        b0 : array (3,)
            Offset vector
        """
        self.A_v = A_v
        self.A_omega = A_omega
        self.b0 = b0

    def state_transition(self, x, dt):
        """
        State transition: constant Velocity model.

        Parameters:
        -----------
        x : array (3,)
            State [Vx, Vy, Vz]
        dt : float
            Time step

        Returns:
        --------
        x_next : array (3,)
            Next state (unchanged for constant Velocity)
        """
        return x.copy()

    def measurement_function(self, x, omega=None):
        """
        Measurement function: map velocity to expected magnetic field.

        Uses linear model: b = A_v * v + A_omega * omega + b0

        Parameters:
        -----------
        x : array (3,)
            State [Vx, Vy, Vz]
        omega : array (3,) or None
            Angular velocity (rad/s)

        Returns:
        --------
        z : array (3,)
            Expected measurement [Bx, By, Bz]
        """
        if omega is None:
            omega = self.current_omega

        V = x
        b_pred = self.A_v @ V + self.A_omega @ omega + self.b0
        return b_pred

    def predict(self):
        """UKF prediction step."""
        self.ukf.predict()

    def update(self, Bx, By, Bz, omega):
        """
        UKF update step with magnetic field measurement.

        Parameters:
        -----------
        Bx, By, Bz : float
            Magnetic field measurements (Crazyflie frame)
        omega : array (3,)
            Angular velocity [omega_x, omega_y, omega_z] (rad/s)
        """
        self.current_omega = omega
        z = np.array([Bx, By, Bz])
        self.ukf.update(z, omega=omega)

    def get_Velocity(self):
        """Get current Velocity estimate."""
        return self.ukf.x.copy()

    def get_coVariance(self):
        """Get current state coVariance."""
        return self.ukf.P.copy()


def learn_mapping(data):
    """
    Learn linear mapping from velocity AND angular velocity to magnetic field.
    Uses least squares: b = A_v * v + A_omega * omega (NO BIAS TERM)
    """
    print("\n" + "=" * 70)
    print("LEARNING VELOCITY + OMEGA TO MAGNETIC FIELD MAPPING (No Bias)")
    print("=" * 70)

    has_omega = all(col in data.columns for col in ['omega_x', 'omega_y', 'omega_z'])

    V = data[['Vx', 'Vy', 'Vz']].values  # (N, 3)
    B = data[['Bx_cf', 'By_cf', 'Bz_cf']].values  # (N, 3)

    if has_omega:
        Omega = data[['omega_x', 'omega_y', 'omega_z']].values  # (N, 3)
        print(f"Using angular velocity in mapping")

        # NO BIAS TERM - just [v, omega]
        X = np.hstack([V, Omega])  # (N, 6) instead of (N, 7)

        # Solve: B = X @ [A_v | A_omega]^T  (no bias)
        params = np.linalg.lstsq(X, B, rcond=None)[0]  # (6, 3)

        A_v = params[:3, :].T  # (3, 3)
        A_omega = params[3:6, :].T  # (3, 3)
        b0 = np.zeros(3)  # NO BIAS

        # Evaluate fit quality
        B_pred = V @ A_v.T + Omega @ A_omega.T  # No +b0
    else:
        print("Warning: No angular velocity data found")

        # NO BIAS TERM - just [v]
        params = np.linalg.lstsq(V, B, rcond=None)[0]  # (3, 3)

        A_v = params.T  # (3, 3)
        A_omega = np.zeros((3, 3))
        b0 = np.zeros(3)  # NO BIAS

        B_pred = V @ A_v.T

    mse = mean_squared_error(B, B_pred)
    r2 = r2_score(B.flatten(), B_pred.flatten())

    print(f"\nMapping quality (no bias):")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")

    print(f"\nVelocity mapping matrix A_v:")
    print(A_v)

    if has_omega:
        print(f"\nAngular velocity mapping matrix A_omega:")
        print(A_omega)

    print(f"\nb0 (forced to zero): {b0}")

    return A_v, A_omega, b0


def run_ukf(data, A_v, A_omega, b0):
    """
    Run UKF on the data.

    Parameters:
    -----------
    data : DataFrame
        Data with time, magnetic field, velocities, omega, etc.
    A_v : array (3, 3)
        Velocity mapping matrix
    A_omega : array (3, 3)
        Angular velocity mapping matrix
    b0 : array (3,)
        Offset vector

    Returns:
    --------
    results : dict
        Results with estimated and ground truth Velocities
    """
    print("\n" + "="*70)
    print("RUNNING UKF FOR VELOCITY ESTIMATION")
    print("="*70)

    # Determine time step
    if 'time' in data.columns:
        time = data['time'].values
        dt = np.median(np.diff(time))
    elif 'timestamp' in data.columns:
        time = data['timestamp'].values
        dt = np.median(np.diff(time))
    else:
        dt = 0.02
        time = np.arange(len(data)) * dt

    if time[0] > 1000:
        time = time / 1000.0

    print(f"Time step: {dt:.4f} s ({1/dt:.1f} Hz)")

    # Check if omega is available
    has_omega = all(col in data.columns for col in ['omega_x', 'omega_y', 'omega_z'])
    if not has_omega:
        print("Warning: No angular velocity data, using zeros")

    # Initialize filter
    Velocity_filter = VelocityUKF(dt=dt, process_noise=0.01, measurement_noise=8)
    Velocity_filter.set_mapping_params(A_v, A_omega, b0)

    # Initialize with first Velocity measurement if available
    if 'Vx' in data.columns:
        Velocity_filter.ukf.x = data[['Vx', 'Vy', 'Vz']].iloc[0].values

    # Storage
    results = {
        'time': time,
        'Vx_est': np.zeros(len(data)),
        'Vy_est': np.zeros(len(data)),
        'Vz_est': np.zeros(len(data)),
        'Vx_true': data['Vx'].values if 'Vx' in data.columns else np.zeros(len(data)),
        'Vy_true': data['Vy'].values if 'Vy' in data.columns else np.zeros(len(data)),
        'Vz_true': data['Vz'].values if 'Vz' in data.columns else np.zeros(len(data)),
        'Bx': data['Bx_cf'].values,
        'By': data['By_cf'].values,
        'Bz': data['Bz_cf'].values,
        'coVariance': np.zeros(len(data))
    }

    if has_omega:
        results['omega_x'] = data['omega_x'].values
        results['omega_y'] = data['omega_y'].values
        results['omega_z'] = data['omega_z'].values

    # Run filter
    print("Processing data...")
    for i in range(len(data)):
        # Prediction step
        Velocity_filter.predict()

        # Get angular velocity
        if has_omega:
            omega = np.array([data['omega_x'].iloc[i],
                            data['omega_y'].iloc[i],
                            data['omega_z'].iloc[i]])
        else:
            omega = np.zeros(3)

        # Update with magnetic field measurement
        Bx = data['Bx_cf'].iloc[i]
        By = data['By_cf'].iloc[i]
        Bz = data['Bz_cf'].iloc[i]
        Velocity_filter.update(Bx, By, Bz, omega)

        # Store results
        V_est = Velocity_filter.get_Velocity()
        results['Vx_est'][i] = V_est[0]
        results['Vy_est'][i] = V_est[1]
        results['Vz_est'][i] = V_est[2]
        results['coVariance'][i] = np.trace(Velocity_filter.get_coVariance())

        if i % 100 == 0:
            print(f"  Progress: {i}/{len(data)} ({100*i/len(data):.1f}%)")

    print("UKF complete!")

    # Evaluate performance
    if 'Vx' in data.columns:
        print("\n" + "="*70)
        print("PERFORMANCE METRICS")
        print("="*70)

        for comp in ['x', 'y', 'z']:
            true = results[f'V{comp}_true']
            est = results[f'V{comp}_est']
            rmse = np.sqrt(mean_squared_error(true, est))
            mae = np.mean(np.abs(true - est))
            r2 = r2_score(true, est)
            print(f"\nV{comp}:")
            print(f"  RMSE: {rmse:.4f} m/s")
            print(f"  MAE: {mae:.4f} m/s")
            print(f"  R²: {r2:.4f}")

        # Overall Velocity magnitude
        V_mag_true = np.sqrt(results['Vx_true']**2 + results['Vy_true']**2 + results['Vz_true']**2)
        V_mag_est = np.sqrt(results['Vx_est']**2 + results['Vy_est']**2 + results['Vz_est']**2)
        rmse_mag = np.sqrt(mean_squared_error(V_mag_true, V_mag_est))
        r2_mag = r2_score(V_mag_true, V_mag_est)

        print(f"\nVelocity Magnitude:")
        print(f"  RMSE: {rmse_mag:.4f} m/s")
        print(f"  R²: {r2_mag:.4f}")

    return results


def plot_results(results, save_path='Velocity_ukf_results.png'):
    """Plot UKF results."""
    print("\nGenerating plots...")

    time = results['time']
    if time[0] > 1000:
        time = time / 1000.0
    time = time - time[0]

    has_omega = 'omega_x' in results
    n_plots = 3

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3*n_plots))

    # Plot 1: Velocity components comparison
    ax = axes[0]
    ax.set_ylim([-3, 3])
    ax.set_xlim([0, 90])
    ax.plot(time, results['Vx_true'], 'b-', linewidth=2, label='Vx True', alpha=0.7)
    ax.plot(time, results['Vx_est'], 'b--', linewidth=2, label='Vx Estimated')
    ax.plot(time, results['Vy_true'], 'r-', linewidth=2, label='Vy True', alpha=0.7)
    ax.plot(time, results['Vy_est'], 'r--', linewidth=2, label='Vy Estimated')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Velocity (m/s)', fontsize=11)
    ax.set_title('Velocity Components (X, Y) - True Vs Estimated', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


    # Plot 3: Velocity magnitude
    ax = axes[1]
    ax.set_ylim([0, 3])
    ax.set_xlim([0, 90])
    V_mag_true = np.sqrt(results['Vx_true']**2 + results['Vy_true']**2 + results['Vz_true']**2)
    V_mag_est = np.sqrt(results['Vx_est']**2 + results['Vy_est']**2 + results['Vz_est']**2)
    ax.plot(time, V_mag_true, 'b-', linewidth=2.5, label='True', alpha=0.7)
    ax.plot(time, V_mag_est, 'r--', linewidth=2, label='Estimated')
    #ax.fill_between(time, 0, V_mag_true, alpha=0.2, color='blue')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Speed (m/s)', fontsize=11)
    ax.set_title('Velocity Magnitude - True Vs Estimated', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Magnetic field measurements
    ax = axes[2]
    ax.set_xlim([0, 90])
    ax.set_ylim([-100,100])
    ax.plot(time, results['Bx'], label='Bx', linewidth=2)
    ax.plot(time, results['By'], label='By', linewidth=2)
    ax.plot(time, results['Bz'], label='Bz', linewidth=2)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Magnetic Field', fontsize=11)
    ax.set_title('Magnetic Field Measurements (Crazyflie Frame)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)



    plt.tight_layout()
    #plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {save_path}")
    plt.show()


def main():
    """Main execution."""

    filepath = r"C:\Users\ltjth\Documents\Research\UKF_Data\squareLog1.0.csv"

    print("="*70)
    print("VELOCITY ESTIMATION UKF (with Angular Velocity)")
    print("="*70)
    print(f"Data file: {filepath}\n")

    # Load data
    data = load_data(filepath)

    # Learn Velocity + omega to magnetic field mapping
    A_v, A_omega, b0 = learn_mapping(data)

    # Run UKF
    results = run_ukf(data, A_v, A_omega, b0)

    # Plot results
    plot_results(results, save_path='Velocity_ukf_results.png')

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("Generated: Velocity_ukf_results.png")
    print("="*70)


if __name__ == "__main__":
    main()
"""
Complete UKF for Velocity estimation with Interactive Parameter Tuning
Includes angular velocity (omega) in measurement model
"""
