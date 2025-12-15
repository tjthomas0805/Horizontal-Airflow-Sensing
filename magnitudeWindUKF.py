"""
Wind Comparison Analysis - Magnitude Only
Runs UKF on two datasets (with and without wind) and compares results.
Estimates airflow magnitude directly (not components).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.signal import butter, filtfilt


# =============================================================================
# Data Loading
# =============================================================================

def lowpass_filter(data, cutoff=5.0, fs=100.0, order=4):
    """Apply Butterworth low-pass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, data)


def load_data(filepath):
    """Load flight data CSV and preprocess magnetic field measurements."""
    df = pd.read_csv(filepath)

    # Filter magnetic field data
    for axis in ['Bx', 'By']:
        df[f'{axis}_filt'] = lowpass_filter(df[axis].values)

    # Use filtered values
    df['Bx_cf'] = df['Bx_filt']
    df['By_cf'] = df['By_filt']

    # Compute magnitudes
    df['B_mag'] = np.sqrt(df['Bx_cf'] ** 2 + df['By_cf'] ** 2)

    # Compute velocity magnitude
    if 'Vx' in df.columns:
        df['V_mag'] = np.sqrt(df['Vx'] ** 2 + df['Vy'] ** 2)

    print(f"Loaded {len(df)} samples")
    print(f"B_mag range: [{df['B_mag'].min():.1f}, {df['B_mag'].max():.1f}]")
    if 'V_mag' in df.columns:
        print(f"V_mag range: [{df['V_mag'].min():.3f}, {df['V_mag'].max():.3f}] m/s")

    return df


# =============================================================================
# Sensor Model
# =============================================================================

class QuadraticSensorModel:
    """
    Quadratic sensor model mapping airflow velocity magnitude to magnetic field magnitude.
    """

    def __init__(self):
        # Inverse model coefficients: B_mag -> v_mag
        self.inv_a = -0.000213
        self.inv_b = 0.03955
        self.inv_c = 0.03621

        # Forward model coefficients: v_mag -> B_mag
        self.fwd_a = 29.47
        self.fwd_b = -7.7
        self.fwd_c = 5.44

    def velocity_to_B(self, v_mag):
        """Predict B magnitude from airflow speed."""
        B = self.fwd_a * v_mag ** 2 + self.fwd_b * v_mag + self.fwd_c
        return np.maximum(B, 0)

    def B_to_velocity(self, B_mag):
        """Estimate airflow speed from B magnitude."""
        v = self.inv_a * B_mag ** 2 + self.inv_b * B_mag + self.inv_c
        return np.maximum(v, 0)


# =============================================================================
# Airflow Magnitude UKF (1D State)
# =============================================================================

class AirflowMagnitudeUKF:
    """
    UKF estimating relative airflow magnitude only.

    State: [va_mag] - airflow speed magnitude
    Measurement: [B_mag] - magnetic field magnitude
    """

    def __init__(self, dt, sensor_model, Q_airflow=0.5, R_magnetic=None, R_odom=0.003):
        self.dt = dt
        self.sensor_model = sensor_model
        self.R_odom = R_odom

        # State dimension
        self.dim_x = 1  # [va_mag]
        self.dim_z = 1  # [B_mag]

        # Sigma points
        points = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.1, beta=2.0, kappa=0.0)

        # Create UKF
        self.ukf = UKF(
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            dt=dt,
            fx=self._fx,
            hx=self._hx,
            points=points
        )

        # Initial state: zero airflow
        self.ukf.x = np.array([0.0])

        # Initial covariance
        self.ukf.P = np.array([[1.0]])

        # Process noise
        self.ukf.Q = np.array([[Q_airflow]])

        # Measurement noise
        if R_magnetic is not None:
            self.ukf.R = np.array([[R_magnetic]])
        else:
            # Default: approximate from your 2x2 matrix (use average variance)
            self.ukf.R = np.array([[0.74]])

    def _fx(self, x, dt):
        """Process model: random walk on magnitude."""
        return x

    def _hx(self, x):
        """Measurement model: v_mag -> B_mag using quadratic model."""
        va_mag = np.maximum(x[0], 0)  # Ensure non-negative

        # B_mag = a*v^2 + b*v + c
        a = 29.47
        b = -7.7
        c = 5.44

        B_mag = a * va_mag ** 2 + b * va_mag + c
        return np.array([B_mag])

    def predict(self):
        """Prediction step."""
        self.ukf.P = 0.5 * (self.ukf.P + self.ukf.P.T)
        self.ukf.P += np.eye(self.dim_x) * 1e-6
        self.ukf.predict()

        # Ensure non-negative after predict
        self.ukf.x[0] = max(self.ukf.x[0], 0)

    def update_with_velocity(self, v_mag):
        """Soft constraint from odometry velocity magnitude."""
        z = np.array([v_mag])
        H = np.array([[1.0]])
        R_odom = np.array([[self.R_odom]])

        y = z - self.ukf.x
        S = H @ self.ukf.P @ H.T + R_odom
        K = self.ukf.P @ H.T / S[0, 0]

        self.ukf.x = self.ukf.x + K.flatten() * y[0]
        self.ukf.P = (np.eye(1) - K @ H) @ self.ukf.P

        # Ensure non-negative
        self.ukf.x[0] = max(self.ukf.x[0], 0)

    def update(self, B_mag):
        """Measurement update with magnetic field magnitude."""
        z = np.array([B_mag])

        self.ukf.P = 0.5 * (self.ukf.P + self.ukf.P.T)
        self.ukf.P += np.eye(self.dim_x) * 1e-6

        try:
            self.ukf.update(z)
        except np.linalg.LinAlgError:
            self.ukf.P = np.array([[1.0]])

        # Ensure non-negative
        self.ukf.x[0] = max(self.ukf.x[0], 0)

    @property
    def airflow_mag(self):
        return self.ukf.x[0]


# =============================================================================
# Run UKF
# =============================================================================

def run_ukf(df, Q_airflow=0.005, R_magnetic=None, R_odom=0.003):
    """Run magnitude-only UKF on a dataset and return results."""

    # Determine timestep
    if 'time' in df.columns:
        time = df['time'].values
        if time[0] > 1000:
            time = time / 1000.0
        dt = np.median(np.diff(time))
    else:
        dt = 0.02
        time = np.arange(len(df)) * dt

    time = time - time[0]

    # Initialize
    sensor_model = QuadraticSensorModel()
    ukf = AirflowMagnitudeUKF(dt, sensor_model, Q_airflow=Q_airflow,
                              R_magnetic=R_magnetic, R_odom=R_odom)

    N = len(df)
    results = {
        'time': time,
        'va_mag': np.zeros(N),
        'v_empirical': np.zeros(N),
        'v_true_mag': df['V_mag'].values,
        'B_mag': df['B_mag'].values,
        'wind_mag': np.zeros(N),
    }

    # Run filter
    for i in range(N):
        ukf.predict()

        # Update with B magnitude
        B_mag = df['B_mag'].iloc[i]
        ukf.update(B_mag)

        # Update with velocity magnitude
        v_mag = df['V_mag'].iloc[i]
        ukf.update_with_velocity(v_mag)

        results['va_mag'][i] = ukf.airflow_mag
        results['v_empirical'][i] = sensor_model.B_to_velocity(B_mag)

        # Wind magnitude = |airflow - velocity|
        # In no-wind case, airflow ≈ velocity, so wind ≈ 0
        results['wind_mag'][i] = abs(ukf.airflow_mag - v_mag)

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_comparison(results_no_wind, results_wind, title_no_wind="No Wind", title_wind="With Wind"):
    """Create 2x2 comparison plot."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: No wind - velocity comparison
    ax = axes[0, 0]
    ax.plot(results_no_wind['time'], results_no_wind['v_true_mag'],
            'b-', lw=2, label='State Estimator', alpha=0.8)
    ax.plot(results_no_wind['time'], results_no_wind['va_mag'],
            'r-', lw=2, label='UKF', alpha=0.8)
    ax.plot(results_no_wind['time'], results_no_wind['v_empirical'],
            color='grey', lw=1, label='Model', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_ylim([0, 2])
    ax.set_title(f'{title_no_wind}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top right: With wind - velocity comparison
    ax = axes[0, 1]
    ax.plot(results_wind['time'], results_wind['v_true_mag'],
            'b-', lw=2, label='State Estimator', alpha=0.8)
    ax.plot(results_wind['time'], results_wind['va_mag'],
            'r-', lw=2, label='UKF', alpha=0.8)
    ax.plot(results_wind['time'], results_wind['v_empirical'],
            color='grey', lw=1, label='Model', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_ylim([0, 2])
    ax.set_title(f'{title_wind}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom left: No wind - wind estimate (should be ~0)
    ax = axes[1, 0]
    ax.plot(results_no_wind['time'], results_no_wind['wind_mag'],
            'k-', lw=2, alpha=0.8)
    ax.axhline(0.18, color='gray', label='Threshold', ls='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylim([0, 1.25])
    ax.set_ylabel('Wind (m/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom right: With wind - wind estimate
    ax = axes[1, 1]
    ax.plot(results_wind['time'], results_wind['wind_mag'],
            'k-', lw=2, alpha=0.8)
    ax.axhline(0.18, color='gray', label='Threshold', ls='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylim([0, 1.25])
    ax.set_ylabel('Wind (m/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('wind_comparison_magnitude.png', dpi=150, bbox_inches='tight')
    print("Saved: wind_comparison_magnitude.png")

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    # =========================================================================
    # UPDATE THESE FILE PATHS
    # =========================================================================
    no_wind_file = r"C:\Users\ltjth\Documents\Research\UKF_Data\CF_ARR_Line_EKF2.0ms_4.5m2.csv"
    wind_file = r"C:\Users\ltjth\Documents\Research\UKF_Data\CF_ARR_Line_EKF2.0ms_4.5m_FAN1.csv"

    print("=" * 60)
    print("WIND COMPARISON ANALYSIS - MAGNITUDE ONLY")
    print("=" * 60)

    # UKF parameters
    Q_airflow = 0.005
    R_odom = 0.003
    R_magnetic = 0.74  # Scalar for 1D measurement

    # Load data
    print(f"\nLoading no-wind file...")
    print(f"  {no_wind_file}")
    df_no_wind = load_data(no_wind_file)

    print(f"\nLoading wind file...")
    print(f"  {wind_file}")
    df_wind = load_data(wind_file)

    # Run UKF on both
    print("\nRunning UKF on no-wind data...")
    results_no_wind = run_ukf(df_no_wind, Q_airflow=Q_airflow,
                              R_magnetic=R_magnetic, R_odom=R_odom)

    print("Running UKF on wind data...")
    results_wind = run_ukf(df_wind, Q_airflow=Q_airflow,
                           R_magnetic=R_magnetic, R_odom=R_odom)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nNo Wind:")
    print(f"  Wind estimate: {np.mean(results_no_wind['wind_mag']):.3f} ± {np.std(results_no_wind['wind_mag']):.3f} m/s")
    print(f"\nWith Wind:")
    print(f"  Wind estimate: {np.mean(results_wind['wind_mag']):.3f} ± {np.std(results_wind['wind_mag']):.3f} m/s")

    # Plot comparison
    print("\nGenerating comparison plot...")
    plot_comparison(results_no_wind, results_wind,
                    title_no_wind="No Wind", title_wind="With Wind (Fan)")

    plt.show()

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
