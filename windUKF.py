"""
UKF for Relative Airflow Estimation from Magnetic Whisker Sensor

Estimates: [va_x, va_y] - relative airflow velocity at the sensor in body frame
Uses drone odometry (velocity, acceleration, angular velocity) in the process model
to predict how airflow changes during maneuvers, preventing false readings during transients.

Wind is computed separately via: v_wind = v_airflow + v_drone

Sensor model (quadratic, symmetric for x and y):
    B = f(v_airflow)  where  |B| = a*|v|^2 + b*|v| + c
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R


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
    for axis in ['Bx', 'By']:#, 'Bz']:
        df[f'{axis}_filt'] = lowpass_filter(df[axis].values)

    # Use filtered values (keeping cf suffix for compatibility)
    df['Bx_cf'] = df['Bx_filt']
    df['By_cf'] = df['By_filt']
    #df['Bz_cf'] = df['Bz_filt']

    # Compute magnitudes
    df['B_mag'] = np.sqrt(df['Bx_cf']**2 + df['By_cf']**2)# + df['Bz_cf']**2)
    df['Bxy_mag'] = np.sqrt(df['Bx_cf']**2 + df['By_cf']**2)

    # Compute qw if quaternion components exist
    if all(col in df.columns for col in ['Qx', 'Qy', 'Qz']):
        qw_sq = 1.0 - (df['Qx']**2 + df['Qy']**2 + df['Qz']**2)
        df['qw'] = np.sqrt(np.maximum(qw_sq, 0))

    # Compute velocity magnitude for reference
    if 'Vx' in df.columns:
        df['V_mag'] = np.sqrt(df['Vx']**2 + df['Vy']**2)# + df['Vz']**2)

    print(f"Loaded {len(df)} samples")
    print(f"B range: [{df['B_mag'].min():.1f}, {df['B_mag'].max():.1f}]")
    if 'V_mag' in df.columns:
        print(f"V range: [{df['V_mag'].min():.3f}, {df['V_mag'].max():.3f}] m/s")

    return df


# =============================================================================
# Sensor Model
# =============================================================================

class QuadraticSensorModel:
    """
    Quadratic sensor model mapping airflow velocity to magnetic field.

    Forward:  v -> B_mag  using  B = a*v^2 + b*v + c
    Inverse:  B -> v      using  v = a'*B^2 + b'*B + c'

    Assumes symmetric response in x and y axes.
    """

    def __init__(self):
        # Inverse model coefficients: B_mag -> v_mag
        # v = a*B^2 + b*B + c
        self.inv_a = -0.0001523
        self.inv_b = 0.03312
        self.inv_c = 0.1211

        # Forward model coefficients: v_mag -> B_mag
        # B = a*v^2 + b*v + c
        self.fwd_a = 29.47
        self.fwd_b = -7.7
        self.fwd_c = 5.44

    def velocity_to_B(self, v_mag):
        """Predict B magnitude from airflow speed."""
        B = self.fwd_a * v_mag**2 + self.fwd_b * v_mag + self.fwd_c
        return np.maximum(B, 0)

    def B_to_velocity(self, B_mag):
        """Estimate airflow speed from B magnitude (for comparison/init)."""
        v = self.inv_a * B_mag**2 + self.inv_b * B_mag + self.inv_c
        return np.maximum(v, 0)

    def predict_Bxy(self, va_x, va_y):
        """
        Predict [Bx, By] from airflow velocity components.

        Uses magnitude-based model with direction preserved:
        - Compute expected |B| from |v|
        - Scale to get Bx, By proportional to vx, vy
        """
        v_mag = np.sqrt(va_x**2 + va_y**2)

        if v_mag < 1e-6:
            return np.array([self.fwd_c, self.fwd_c])  # baseline when stationary

        B_mag = self.velocity_to_B(v_mag)

        # Distribute B magnitude according to velocity direction
        # This assumes Bx/By respond proportionally to vx/vy
        Bx = B_mag * (va_x / v_mag)
        By = B_mag * (va_y / v_mag)

        return np.array([Bx, By])


# =============================================================================
# Airflow Estimation UKF
# =============================================================================

class AirflowUKF:
    """
    UKF estimating relative airflow velocity in body frame.

    State: [va_x, va_y] - relative airflow at sensor (body frame)

    Process model: Airflow changes when drone accelerates
        va_new = va_old - accel_body * dt
        (If drone accelerates forward, headwind increases)

    Measurement: [Bx, By] from magnetic whisker sensor
        Uses quadratic model with symmetry assumption
    """

    def __init__(self, dt, sensor_model, Q_airflow=0.5, R_magnetic=None):
        """
        Parameters
        ----------
        dt : float
            Time step in seconds
        sensor_model : QuadraticSensorModel
            Sensor model for measurement function
        Q_airflow : float
            Process noise for airflow states. Higher = trust measurements more,
            faster adaptation. Lower = smoother estimates.
        R_magnetic : ndarray (2,2) or None
            Measurement noise covariance. From sensor characterization.
        """
        self.dt = dt
        self.sensor_model = sensor_model

        # State dimension
        self.dim_x = 2  # [va_x, va_y]
        self.dim_z = 2  # [Bx, By]

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
        self.ukf.x = np.zeros(2)

        # Initial covariance
        self.ukf.P = np.eye(2) * 1.0

        # Process noise
        self.ukf.Q = np.eye(2) * Q_airflow

        # Measurement noise (from sensor characterization)
        if R_magnetic is not None:
            self.ukf.R = R_magnetic
        else:
            # Default values
            self.ukf.R = np.array([
                [5.05, 1.88],
                [1.88, 6.64]
            ])

        # Store inputs for process model
        self._accel_body = np.zeros(2)

    def _fx(self, x, dt):
        """
        Process model: predict airflow change from drone acceleration.

        Relative airflow = wind - drone_velocity
        If drone accelerates, drone_velocity increases, so airflow decreases
        in that direction.
        """
        # x = [va_x, va_y]
        # When drone accelerates in +x, airflow in +x decreases
        x_new = x - self._accel_body * dt
        return x

    def _hx(self,x):
        va_x, va_y = x
        v_mag = np.sqrt(va_x ** 2 + va_y ** 2)

        c = 29.47
        b = -7.7

        Bx = c * v_mag * va_x + b * va_x
        By =- c * v_mag * va_y + b * va_y

        return np.array([Bx, By])

    def predict(self, accel_body_xy):
        """
        Prediction step with drone acceleration as input.

        Parameters
        ----------
        accel_body_xy : array-like (2,)
            Drone acceleration in body frame [ax, ay] in m/s^2
        """
        self._accel_body = np.asarray(accel_body_xy)

        # Ensure P stays positive definite
        self.ukf.P = 0.5 * (self.ukf.P + self.ukf.P.T)
        self.ukf.P += np.eye(self.dim_x) * 1e-6

        self.ukf.predict()

    def update_with_velocity(self, vx_body, vy_body):
        """
        Soft constraint: for indoor/no-wind, airflow ≈ -velocity
        High R = weak constraint = whisker still dominates
        """
        # Expected airflow if no wind
        z = np.array([-vx_body, -vy_body])

        H = np.eye(2)

        # HIGH uncertainty = weak pull toward odometry
        # Tune this: bigger = weaker constraint = more whisker influence
        R_odom = np.eye(2) *0.003 # start here, increase if whisker gets drowned out

        y = z - self.ukf.x
        S = H @ self.ukf.P @ H.T + R_odom
        K = self.ukf.P @ H.T @ np.linalg.inv(S)

        self.ukf.x = self.ukf.x + K @ y
        self.ukf.P = (np.eye(2) - K @ H) @ self.ukf.P

    def update(self, Bx, By):
        """
        Measurement update with magnetic field readings.

        Parameters
        ----------
        Bx, By : float
            Filtered magnetic field components
        """
        z = np.array([Bx, By])

        # Ensure P stays positive definite
        self.ukf.P = 0.5 * (self.ukf.P + self.ukf.P.T)
        self.ukf.P += np.eye(self.dim_x) * 1e-6

        try:
            self.ukf.update(z)
        except np.linalg.LinAlgError:
            # Reset covariance if things go wrong
            self.ukf.P = np.eye(self.dim_x) * 1.0

    @property
    def airflow(self):
        """Current airflow estimate [va_x, va_y]."""
        return self.ukf.x.copy()

    @property
    def covariance(self):
        """Current state covariance."""
        return self.ukf.P.copy()


# =============================================================================
# Main Processing
# =============================================================================

def compute_body_acceleration(df, dt):
    """
    Compute body-frame acceleration from velocity and quaternion.

    Returns array of shape (N, 2) with [ax, ay] for each timestep.
    """
    N = len(df)
    accel_body = np.zeros((N, 2))

    # Get world-frame velocities
    vx = df['Vx'].values
    vy = df['Vy'].values
   #vz = df['Vz'].values

    # Compute world-frame acceleration (finite difference)
    ax_world = np.gradient(vx, dt)
    ay_world = np.gradient(vy, dt)
   # az_world = np.gradient(vz, dt)

    # Smooth the acceleration
    ax_world = lowpass_filter(ax_world, cutoff=10.0, fs=1/dt)
    ay_world = lowpass_filter(ay_world, cutoff=10.0, fs=1/dt)

    # Transform to body frame
    for i in range(N):
        qw = df['qw'].iloc[i]
        qx = df['Qx'].iloc[i]
        qy = df['Qy'].iloc[i]
        qz = df['Qz'].iloc[i]

        # scipy uses [x, y, z, w] convention
        rot = R.from_quat([qx, qy, qz, qw])
        R_world_to_body = rot.inv().as_matrix()

        a_world = np.array([ax_world[i], ay_world[i],0])
        a_body = R_world_to_body @ a_world

        accel_body[i, 0] = a_body[0]
        accel_body[i, 1] = a_body[1]

    return accel_body


def run_airflow_ukf(df, Q_airflow=0.5, R_magnetic=None):
    """
    Run the airflow estimation UKF on flight data.

    Parameters
    ----------
    df : DataFrame
        Flight data with columns: Bx_cf, By_cf, Vx, Vy, Vz, qw, qx, qy, qz
    Q_airflow : float
        Process noise for airflow. Tune this for responsiveness vs smoothness.
    R_magnetic : ndarray (2,2) or None
        Measurement noise covariance from sensor characterization.

    Returns
    -------
    results : dict
        Estimation results including airflow, empirical velocity, and errors.
    """
    # Determine timestep
    if 'time' in df.columns:
        time = df['time'].values
        if time[0] > 1000:  # milliseconds
            time = time / 1000.0
        dt = np.median(np.diff(time))
    else:
        dt = 0.02
        time = np.arange(len(df)) * dt

    time = time - time[0]  # start from zero

    print(f"Timestep: {dt:.4f}s ({1/dt:.1f} Hz)")
    print(f"Q_airflow: {Q_airflow}")

    # Initialize
    sensor_model = QuadraticSensorModel()
    ukf = AirflowUKF(dt, sensor_model, Q_airflow=Q_airflow, R_magnetic=R_magnetic)

    # Compute body-frame acceleration for process model
    accel_body = compute_body_acceleration(df, dt)

    # Storage
    N = len(df)
    results = {
        'time': time,
        # Airflow estimates
        'va_x': np.zeros(N),
        'va_y': np.zeros(N),
        'va_mag': np.zeros(N),
        # Empirical model (direct inversion for comparison)
        'v_empirical': np.zeros(N),
        # Ground truth velocity (for comparison)
        'vx_true': df['Vx'].values,
        'vy_true': df['Vy'].values,
        #'vz_true': df['Vz'].values,
        # Magnetic field
        'Bx': df['Bx_cf'].values,
        'By': df['By_cf'].values,
        #'Bz': df['Bz_cf'].values,
        'B_mag': df['B_mag'].values,
        # Wind estimate (computed from airflow + drone velocity)
        'wind_x': np.zeros(N),
        'wind_y': np.zeros(N),
        'wind_mag': np.zeros(N),
    }

    # Run filter
    print("Running UKF...")
    for i in range(N):
        # Predict with acceleration input
        ukf.predict(accel_body[i])

        # Update with magnetic measurement
        Bx = df['Bx_cf'].iloc[i]
        By = df['By_cf'].iloc[i]
        ukf.update(Bx, By)
        ukf.update_with_velocity(df['Vx'].iloc[i], df['Vy'].iloc[i])
        # Store airflow estimate
        va = ukf.airflow
        results['va_x'][i] = va[0]
        results['va_y'][i] = va[1]
        results['va_mag'][i] = np.sqrt(va[0]**2 + va[1]**2)

        # Empirical model for comparison
        B_mag = df['B_mag'].iloc[i]
        results['v_empirical'][i] = sensor_model.B_to_velocity(B_mag)

        # Compute wind: v_wind = v_airflow + v_drone
        # Note: airflow is what sensor sees, drone velocity is in world frame
        # For indoor flight with no wind, airflow ≈ -v_drone (in body frame)
        # This is a simplification - proper wind triangle needs frame transforms
        results['wind_x'][i] = va[0] + df['Vx'].iloc[i]
        results['wind_y'][i] = va[1] + df['Vy'].iloc[i]
        results['wind_mag'][i] = np.sqrt(results['wind_x'][i]**2 + results['wind_y'][i]**2)

    print("Done!")

    # Performance metrics
    v_true_mag = np.sqrt(results['vx_true']**2 + results['vy_true']**2)

    # The airflow magnitude should match velocity magnitude (no wind case)
    rmse_ukf = np.sqrt(np.mean((results['va_mag'] - v_true_mag)**2))
    rmse_emp = np.sqrt(np.mean((results['v_empirical'] - v_true_mag)**2))

    print(f"\nPerformance (vs ground truth velocity magnitude):")
    print(f"  UKF airflow RMSE:      {rmse_ukf:.4f} m/s")
    print(f"  Empirical model RMSE:  {rmse_emp:.4f} m/s")
    print(f"\nWind estimate (should be ~0 indoors):")
    print(f"  Mean: [{np.mean(results['wind_x']):.3f}, {np.mean(results['wind_y']):.3f}] m/s")
    print(f"  Std:  [{np.std(results['wind_x']):.3f}, {np.std(results['wind_y']):.3f}] m/s")

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_results(results):
    """Generate result plots."""
    time = results['time']

    # Compute velocity magnitudes
    v_true_mag = np.sqrt(results['vx_true']**2 + results['vy_true']**2)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # 1. Airflow magnitude vs ground truth velocity
    ax = axes[0, 0]
    ax.plot(time, v_true_mag, 'b-', lw=2, label='Ground Truth |V|', alpha=0.8)
    ax.plot(time, results['va_mag'], 'r-', lw=2, label='UKF |Airflow|', alpha=0.8)
    ax.plot(time, results['v_empirical'], 'g--', lw=1.5, label='Empirical', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Airflow Magnitude vs Ground Truth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. X component
    ax = axes[0, 1]
    ax.plot(time, -results['vx_true'], 'b-', lw=2, label='Ground Truth -Vx', alpha=0.8)
    ax.plot(time, results['va_x'], 'r-', lw=2, label='UKF Airflow X', alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('X Component (airflow ≈ -velocity for no wind)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Y component
    ax = axes[1, 0]
    ax.plot(time, -results['vy_true'], 'b-', lw=2, label='Ground Truth -Vy', alpha=0.8)
    ax.plot(time, results['va_y'], 'r-', lw=2, label='UKF Airflow Y', alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Y Component')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Estimation error
    ax = axes[1, 1]
    error_mag = results['va_mag'] - v_true_mag
    error_emp = results['v_empirical'] - v_true_mag
    ax.plot(time, error_mag, 'r-', lw=1.5, label='UKF Error', alpha=0.8)
    ax.plot(time, error_emp, 'g--', lw=1.5, label='Empirical Error', alpha=0.7)
    ax.axhline(0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (m/s)')
    ax.set_title('Magnitude Estimation Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Wind estimate
    ax = axes[2, 0]
    ax.plot(time, results['wind_x'], 'b-', lw=2, label='Wind X')
    ax.plot(time, results['wind_y'], 'r-', lw=2, label='Wind Y')
    ax.plot(time, results['wind_mag'], 'g--', lw=2, label='|Wind|')
    ax.axhline(0, color='k', ls='--', alpha=0.5)
    ax.fill_between(time, -0.2, 0.2, alpha=0.2, color='gray')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Wind (m/s)')
    ax.set_title('Wind Estimate (should be ~0 indoors)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Magnetic field
    ax = axes[2, 1]
    ax.plot(time, results['Bx'], 'b-', lw=1.5, label='Bx', alpha=0.7)
    ax.plot(time, results['By'], 'r-', lw=1.5, label='By', alpha=0.7)
    ax.plot(time, results['B_mag'], 'g-', lw=1.5, label='|B|', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Magnetic Field')
    ax.set_title('Magnetic Field Measurements')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('airflow_ukf_results.png', dpi=150, bbox_inches='tight')
    print("\nSaved: airflow_ukf_results.png")

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    # Data path
    filepath = r"C:\Users\ltjth\Documents\Research\UKF_Data\CF_ARR_Square_EKF2.0ms_4.5m1.csv"

    print("=" * 60)
    print("AIRFLOW ESTIMATION UKF")
    print("=" * 60)

    # Load data
    df = load_data(filepath)

    # Measurement noise from your sensor characterization
    R_magnetic = np.array([
        [0.72422324, 0.33065],
         [0.33065  ,  0.76509166]
    ])

    # Run UKF
    # Q_airflow is your main tuning knob:
    #   Low (0.1):  Smooth, slow to adapt
    #   Med (0.5):  Balanced
    #   High (2.0): Fast adaptation, more noise
    results = run_airflow_ukf(df, Q_airflow=0.005, R_magnetic=R_magnetic)

    # Plot
    plot_results(results)
    plt.show()

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
