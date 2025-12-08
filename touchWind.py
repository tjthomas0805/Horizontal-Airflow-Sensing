"""
Simplified UKF for Velocity and Wind Estimation
Estimates: [vx, vy, wind_x, wind_y] from magnetic field measurements [Bx, By]

Uses:
- Empirical model: v = -0.0001523*b_mag^2 + 0.03312*b_mag + 0.1211
- Quaternion for coordinate transformations
- Angular velocity (omega) for dynamics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import butter, filtfilt


def lowpass_filter(data, cutoff=5, fs=100, order=4):
    """Apply low-pass Butterworth filter."""
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)


def load_data(filepath):
    """Load and preprocess flight data."""
    df = pd.read_csv(filepath)

    print("\nApplying low-pass filter to magnetic field data (5 Hz cutoff @ 100 Hz)...")
    df['Bx_filt'] = lowpass_filter(df['Bx'].values)
    df['By_filt'] = lowpass_filter(df['By'].values)
    df['Bz_filt'] = lowpass_filter(df['Bz'].values)

    # Crazyflie coordinate frame
    df['Bx_cf'] = df['Bx_filt']
    df['By_cf'] = df['By_filt']
    df['Bz_cf'] = df['Bz_filt']

    df['B_mag'] = np.sqrt(df['Bx_cf']**2 + df['By_cf']**2 + df['Bz_cf']**2)
    df['Bxy_mag'] = np.sqrt(df['Bx_cf']**2 + df['By_cf']**2)

    # Compute qw from quaternion if needed
    if all(col in df.columns for col in ['qx', 'qy', 'qz']):
        print("Computing qw from quaternion components...")
        qx, qy, qz = df['qx'].values, df['qy'].values, df['qz'].values
        qw_squared = 1.0 - (qx**2 + qy**2 + qz**2)
        qw_squared = np.maximum(qw_squared, 0)
        df['qw'] = np.sqrt(qw_squared)

    if 'Vx' in df.columns:
        V_mag = np.sqrt(df['Vx']**2 + df['Vy']**2)
        print(f"\nVelocity magnitude: [{V_mag.min():.3f}, {V_mag.max():.3f}] m/s")

    print(f"\nMagnetic field statistics:")
    print(f"  Bx_cf: [{df['Bx_cf'].min():.2f}, {df['Bx_cf'].max():.2f}]")
    print(f"  By_cf: [{df['By_cf'].min():.2f}, {df['By_cf'].max():.2f}]")
    print(f"  Bz_cf: [{df['Bz_cf'].min():.2f}, {df['Bz_cf'].max():.2f}]")
    print(f"  B_mag: [{df['B_mag'].min():.2f}, {df['B_mag'].max():.2f}]")

    return df


class EmpiricalSensorModel:
    """
    Empirical sensor model: v = a*b_mag^2 + b*b_mag + c
    """

    def __init__(self, a=-0.0001523, b=0.03312, c=0.1211):
        self.a = a
        self.b = b
        self.c = c

        print("\n" + "="*70)
        print("EMPIRICAL SENSOR MODEL")
        print("="*70)
        print(f"Model: v = {a}*b_mag² + {b}*b_mag + {c}")

    def b_mag_to_velocity(self, b_mag):
        """Convert magnetic field magnitude to velocity magnitude."""
        v = self.a * b_mag**2 + self.b * b_mag + self.c
        return np.maximum(v, 0)

    def velocity_to_b_mag(self, v):
        """Convert velocity to expected magnetic field magnitude (inverse)."""
        # discriminant = self.b**2 - 4 * self.a * (self.c - v)
        # discriminant = np.maximum(discriminant, 0)
        b_mag = 29.47*v**2 -7.7*v+5.44
        return np.maximum(b_mag, 0)


class WindEstimationUKF:
    """
    UKF for wind and velocity estimation using direct magnetic field measurements.

    State: [px, py, pz, vx, vy, vz, v_wind_x, v_wind_y]

    Measurements: [Bx, By] - magnetic field components (horizontal only)

    Tuning is done via Q (process noise), not R (measurement noise).
    R values are set from sensor characterization.
    """

    def __init__(self, dt=0.02, calibration=None, empirical_model=None,
                 Q_position=0.01,
                 Q_velocity=0.1,
                 Q_wind=0.01,
                 R_velocity_cov=None,
                 R_magnetic_cov=None,
                 odometry_position_gain=0.6,
                 use_odometry_velocity=True):
        """
        Initialize wind estimation UKF.

        Parameters:
        -----------
        dt : float
            Time step
        calibration : dict
            Calibration parameters (k_x, k_y, k_mag)
        empirical_model : EmpiricalSensorModel
            Empirical sensor model for velocity estimation
        Q_position : float
            Process noise for position states
        Q_velocity : float
            Process noise for velocity states (MAIN TUNING PARAMETER)
            Higher = adapt faster, trust measurements more
            Lower = smoother estimates, trust model more
        Q_wind : float
            Process noise for wind states (SECONDARY TUNING PARAMETER)
            Higher = wind can change quickly
            Lower = assume steady wind
        R_velocity_cov : array (2, 2) or None
            Velocity measurement noise covariance matrix from Crazyflie EKF
            If None, uses default values
        R_magnetic_cov : array (2, 2) or None
            Magnetic sensor measurement noise covariance matrix
            If None, uses default values
        odometry_position_gain : float
            How much to trust odometry position (0 = ignore, 1 = fully trust)
        use_odometry_velocity : bool
            If False, completely ignore odometry velocity updates
        """
        self.dt = dt
        self.empirical_model = empirical_model
        self.odometry_position_gain = odometry_position_gain
        self.use_odometry_velocity = use_odometry_velocity

        # Calibration parameters
        if calibration is None:
            calibration = {'k_x': -10.0, 'k_y': -10.0, 'k_mag': 10.0}
        self.k_x = calibration.get('k_x', -10.0)
        self.k_y = calibration.get('k_y', -10.0)
        self.k_mag = calibration.get('k_mag', 10.0)

        # State: [px, py, pz, vx, vy, vz, v_wind_x, v_wind_y]
        self.dim_x = 8
        self.dim_z = 2  # Measurement: [Bx, By]

        # Drag coefficients
        self.mu1 = 0.20
        self.mu2 = 0.07

        # Create sigma points
        points = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.1, beta=2.0, kappa=0.0)

        # Initialize UKF
        self.ukf = UKF(
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            dt=dt,
            fx=self.state_transition,
            hx=self.measurement_function,
            points=points
        )

        # Initial state: [px, py, pz, vx, vy, vz, wind_x, wind_y]
        self.ukf.x = np.zeros(8)

        # Initial covariance
        self.ukf.P = np.diag([0.5, 0.5, 0.5,  # Position uncertainty
                              2.0, 2.0, 2.0,  # Velocity uncertainty
                              3.0, 3.0])  # Planar wind uncertainty

        # PROCESS NOISE (Q) - THIS IS WHAT YOU TUNE!
        self.ukf.Q = np.diag([
            Q_position, Q_position, Q_position,  # Position process noise
            Q_velocity, Q_velocity, Q_velocity,  # Velocity process noise (MAIN TUNING)
            Q_wind, Q_wind  # Wind process noise (SECONDARY TUNING)
        ])

        # MEASUREMENT NOISE (R) - SET FROM SENSOR SPECS, DON'T TUNE!
        if R_magnetic_cov is not None:
            self.ukf.R = R_magnetic_cov.copy()
        else:
            # Default if not provided (from your measurements)
            self.ukf.R = np.array([[5.0506233, 1.88419894],
                                   [1.88419894, 6.63777725]])

        # Store velocity measurement noise for odometry updates
        if R_velocity_cov is not None:
            self.R_velocity = R_velocity_cov.copy()
        else:
            # Default if not provided (from your measurements)
            self.R_velocity = np.array([[2.83618479e-04, 3.92207351e-05],
                                        [3.92207351e-05, 1.04702344e-04]])

        # Store current orientation
        self.current_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.current_omega = np.zeros(3)
        self.Bz_baseline = 100.0

        print(f"\n{'=' * 70}")
        print("UKF CONFIGURATION")
        print(f"{'=' * 70}")
        print(f"Process Noise (Q) - TUNING PARAMETERS:")
        print(f"  Q_position: {Q_position}")
        print(f"  Q_velocity: {Q_velocity} (MAIN: higher = faster adaptation)")
        print(f"  Q_wind:     {Q_wind} (SECONDARY: higher = variable wind)")
        print(f"\nMeasurement Noise (R) - FROM SENSOR SPECS:")
        print(f"  Magnetic sensor R:\n{self.ukf.R}")
        print(f"  Velocity odometry R:\n{self.R_velocity}")
        print(f"\nOdometry settings:")
        print(f"  Use velocity: {use_odometry_velocity}")
        print(f"  Position gain: {odometry_position_gain}")

    def state_transition(self, x, dt):
        """State transition: simple kinematics."""
        x_next = x.copy()
        x_next[0:3] = x[0:3] + x[3:6] * dt
        return x_next

    def compute_relative_airflow(self, x):
        """Compute relative airflow in body frame."""
        v_world = x[3:6]
        v_wind_world = np.array([x[6], x[7], 0.0])

        quat = self.current_quat
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        R_world_to_body = rot.as_matrix()

        v_inf_world = v_wind_world - v_world
        v_inf_body = R_world_to_body @ v_inf_world

        return v_inf_body

    def measurement_function(self, x, quat=None, omega=None):
        """Measurement function: predict magnetic field components from state."""
        if quat is not None:
            self.current_quat = quat
        if omega is not None:
            self.current_omega = omega

        v_world = x[3:6]
        v_wind_world = np.array([x[6], x[7], 0.0])

        Bx_pred = self.k_x * (v_world[0] - v_wind_world[0])
        By_pred = self.k_y * (v_world[1] - v_wind_world[1])
        return np.array([Bx_pred, By_pred])

    def predict(self, quat, omega):
        """UKF prediction step."""
        self.current_quat = quat
        self.current_omega = omega

        # Ensure P is positive definite
        self.ukf.P = (self.ukf.P + self.ukf.P.T) / 2
        eigenvalues = np.linalg.eigvals(self.ukf.P)
        if np.any(eigenvalues <= 0):
            self.ukf.P = np.diag([0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 3.0, 3.0])
        self.ukf.P = self.ukf.P + np.eye(self.dim_x) * 1e-6

        try:
            self.ukf.predict()
        except np.linalg.LinAlgError:
            self.ukf.P = np.diag([0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 3.0, 3.0])

    def update_odometry(self, position, velocity_body):
        """Update with odometry measurements."""
        if not self.use_odometry_velocity:
            # Only update position if velocity updates disabled
            innovation_pos = position - self.ukf.x[0:3]
            self.ukf.x[0:3] = self.ukf.x[0:3] + self.odometry_position_gain * innovation_pos
            self.ukf.P[0:3, 0:3] *= 0.9
            return

        # Measurement: [px, py, pz, vx, vy, vz]
        z_odom = np.concatenate([position, velocity_body])

        # Measurement matrix H
        H = np.zeros((6, self.dim_x))
        H[0:3, 0:3] = np.eye(3)  # Position
        H[3:6, 3:6] = np.eye(3)  # Velocity

        # Measurement noise covariance (FROM SENSOR SPECS, NOT TUNED!)
        R_odom = np.zeros((6, 6))
        R_odom[0:3, 0:3] = np.eye(3) * 0.01  # Position noise (typically very accurate)
        R_odom[3:5, 3:5] = self.R_velocity  # Vx, Vy from characterization
        R_odom[5, 5] = self.R_velocity[0, 0]  # Vz (assume similar to Vx)

        # Predicted measurement
        z_pred = H @ self.ukf.x

        # Innovation
        y = z_odom - z_pred

        # Innovation covariance
        S = H @ self.ukf.P @ H.T + R_odom

        # Kalman gain
        try:
            K = self.ukf.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: Matrix inversion failed in odometry update")
            self.ukf.x[0:3] += self.odometry_position_gain * (position - self.ukf.x[0:3])
            return

        # State update
        self.ukf.x = self.ukf.x + K @ y

        # Covariance update (Joseph form)
        I_KH = np.eye(self.dim_x) - K @ H
        self.ukf.P = I_KH @ self.ukf.P @ I_KH.T + K @ R_odom @ K.T

        # Ensure symmetry and positive definiteness
        self.ukf.P = (self.ukf.P + self.ukf.P.T) / 2
        self.ukf.P = self.ukf.P + np.eye(self.dim_x) * 1e-6

    def update_magnetic(self, Bx, By, Bz, quat, omega):
        """Update with magnetic field measurement."""
        self.current_quat = quat
        self.current_omega = omega

        # Update Bz baseline
        v_est_mag = np.linalg.norm(self.ukf.x[3:6])
        if v_est_mag < 0.1:
            self.Bz_baseline = 0.95 * self.Bz_baseline + 0.05 * Bz

        # Measurement
        z = np.array([Bx, By])

        # Ensure P is positive definite
        self.ukf.P = (self.ukf.P + self.ukf.P.T) / 2
        eigenvalues = np.linalg.eigvals(self.ukf.P)
        if np.any(eigenvalues <= 0):
            self.ukf.P = np.diag([0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 3.0, 3.0])
        self.ukf.P = self.ukf.P + np.eye(self.dim_x) * 1e-6

        try:
            self.ukf.update(z, quat=quat, omega=omega)
        except np.linalg.LinAlgError as e:
            print(f"Warning: Skipping magnetic update: {e}")

        self.ukf.P = (self.ukf.P + self.ukf.P.T) / 2
        self.ukf.P = self.ukf.P + np.eye(self.dim_x) * 1e-6

    def update_with_empirical_velocity(self, B_mag):
        """Use empirical model to constrain velocity estimate."""
        if self.empirical_model is None:
            return

        v_empirical = self.empirical_model.b_mag_to_velocity(B_mag)
        # Could add soft constraint here if needed

    def get_state(self):
        """Get current state estimate."""
        wind_world = np.array([self.ukf.x[6], self.ukf.x[7], 0.0])
        return {
            'position': self.ukf.x[0:3].copy(),
            'velocity_body': self.ukf.x[3:6].copy(),
            'wind_world': wind_world,
            'v_infinity_body': wind_world - self.ukf.x[3:6].copy(),
            'covariance': self.ukf.P.copy()
        }

    def compute_drag_force(self):
        """Compute drag force from current state."""
        return np.zeros(3)


def run_wind_ukf(data, calibration, empirical_model=None,
                 Q_position=0.01,
                 Q_velocity=0.1,
                 Q_wind=0.01,
                 R_velocity_cov=None,
                 R_magnetic_cov=None,
                 odometry_position_gain=0.6,
                 use_odometry_velocity=True):
    """
    Run the wind estimation UKF with proper Q tuning.

    TUNING GUIDE (Process Noise Q):
    --------------------------------
    Q_velocity (MAIN TUNING PARAMETER):
      - Low (0.01-0.1): Trust model more, smooth estimates, slower adaptation
      - Medium (0.1-1.0): Balanced
      - High (1.0-10.0): Trust measurements more, fast adaptation, track changes

    Q_wind (SECONDARY TUNING PARAMETER):
      - Low (0.001-0.01): Assume steady wind
      - Medium (0.01-0.1): Allow moderate wind changes
      - High (0.1-1.0): Wind can change rapidly

    Q_position:
      - Typically keep low (0.001-0.1)

    R matrices (measurement noise):
      - Should be SET from sensor characterization, NOT tuned!
      - Provide R_velocity_cov and R_magnetic_cov from your measurements
    """
    print("\n" + "=" * 70)
    print("RUNNING WIND ESTIMATION UKF (Q-Based Tuning)")
    print("=" * 70)

    # Determine time step
    if 'time' in data.columns:
        time = data['time'].values
        dt = np.median(np.diff(time))
    else:
        dt = 0.02
        time = np.arange(len(data)) * dt

    if time[0] > 1000:
        time = time / 1000.0

    print(f"Time step: {dt:.4f} s ({1 / dt:.1f} Hz)")

    # Initialize filter
    wind_ukf = WindEstimationUKF(
        dt=dt,
        calibration=calibration,
        empirical_model=empirical_model,
        Q_position=Q_position,
        Q_velocity=Q_velocity,
        Q_wind=Q_wind,
        R_velocity_cov=R_velocity_cov,
        R_magnetic_cov=R_magnetic_cov,
        odometry_position_gain=odometry_position_gain,
        use_odometry_velocity=use_odometry_velocity
    )

    # Initialize Bz baseline
    wind_ukf.Bz_baseline = data['Bz_cf'].iloc[:100].mean()
    print(f"Bz baseline: {wind_ukf.Bz_baseline:.2f}")

    # Initialize state
    if all(col in data.columns for col in ['PosX', 'PosY', 'PosZ']):
        wind_ukf.ukf.x[0:3] = data[['PosX', 'PosY', 'PosZ']].iloc[0].values
    if 'Vx' in data.columns:
        wind_ukf.ukf.x[3:6] = data[['Vx', 'Vy', 'Vz']].iloc[0].values

    # Storage
    results = {
        'time': time,
        'px_est': np.zeros(len(data)),
        'py_est': np.zeros(len(data)),
        'pz_est': np.zeros(len(data)),
        'px_true': data['PosX'].values if 'PosX' in data.columns else np.zeros(len(data)),
        'py_true': data['PosY'].values if 'PosY' in data.columns else np.zeros(len(data)),
        'pz_true': data['PosZ'].values if 'PosZ' in data.columns else np.zeros(len(data)),
        'vx_est': np.zeros(len(data)),
        'vy_est': np.zeros(len(data)),
        'vz_est': np.zeros(len(data)),
        'vx_true': data['Vx'].values if 'Vx' in data.columns else np.zeros(len(data)),
        'vy_true': data['Vy'].values if 'Vy' in data.columns else np.zeros(len(data)),
        'vz_true': data['Vz'].values if 'Vz' in data.columns else np.zeros(len(data)),
        'v_empirical': np.zeros(len(data)),
        'wind_x': np.zeros(len(data)),
        'wind_y': np.zeros(len(data)),
        'wind_z': np.zeros(len(data)),
        'wind_mag': np.zeros(len(data)),
        'drag_x': np.zeros(len(data)),
        'drag_y': np.zeros(len(data)),
        'drag_z': np.zeros(len(data)),
        'Bx': data['Bx_cf'].values,
        'By': data['By_cf'].values,
        'Bz': data['Bz_cf'].values,
        'B_mag': data['B_mag'].values,
    }

    # Run filter
    print("Processing data...")
    for i in range(len(data)):
        # Get orientation
        if all(col in data.columns for col in ['qw', 'qx', 'qy', 'qz']):
            quat = np.array([data['qw'].iloc[i], data['qx'].iloc[i],
                             data['qy'].iloc[i], data['qz'].iloc[i]])
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Get angular velocity
        if all(col in data.columns for col in ['PitchRate', 'RollRate', 'YawRate']):
            omega = np.array([data['PitchRate'].iloc[i],
                              data['RollRate'].iloc[i],
                              data['YawRate'].iloc[i]])
        else:
            omega = np.zeros(3)

        # Prediction
        wind_ukf.predict(quat, omega)

        # Update with odometry
        has_position = all(col in data.columns for col in ['PosX', 'PosY', 'PosZ'])
        if has_position and 'Vx' in data.columns:
            position = np.array([data['PosX'].iloc[i],
                                 data['PosY'].iloc[i],
                                 data['PosZ'].iloc[i]])
            velocity = np.array([data['Vx'].iloc[i],
                                 data['Vy'].iloc[i],
                                 data['Vz'].iloc[i]])
            wind_ukf.update_odometry(position, velocity)

        # Update with magnetic field
        Bx = data['Bx_cf'].iloc[i]
        By = data['By_cf'].iloc[i]
        Bz = data['Bz_cf'].iloc[i]
        wind_ukf.update_magnetic(Bx, By, Bz, quat, omega)

        # Empirical model
        if empirical_model is not None:
            B_mag = data['B_mag'].iloc[i]
            wind_ukf.update_with_empirical_velocity(B_mag)
            results['v_empirical'][i] = empirical_model.b_mag_to_velocity(B_mag)

        # Store results
        state = wind_ukf.get_state()
        results['px_est'][i] = state['position'][0]
        results['py_est'][i] = state['position'][1]
        results['pz_est'][i] = state['position'][2]
        results['vx_est'][i] = state['velocity_body'][0]
        results['vy_est'][i] = state['velocity_body'][1]
        results['vz_est'][i] = state['velocity_body'][2]
        results['wind_x'][i] = state['wind_world'][0]
        results['wind_y'][i] = state['wind_world'][1]
        results['wind_z'][i] = state['wind_world'][2]
        results['wind_mag'][i] = np.linalg.norm(state['wind_world'])


    print("UKF complete!")

    # Print performance
    if 'Vx' in data.columns:
        print("\n" + "=" * 70)
        print("VELOCITY ESTIMATION PERFORMANCE")
        print("=" * 70)
        global v_mag_true
        v_mag_true = np.sqrt(results['vx_true'] ** 2 + results['vy_true'] ** 2 + results['vz_true'] ** 2)
        v_mag_est = np.sqrt(results['vx_est'] ** 2 + results['vy_est'] ** 2 + results['vz_est'] ** 2)

        rmse = np.sqrt(mean_squared_error(v_mag_true, v_mag_est))
        r2 = r2_score(v_mag_true, v_mag_est)

        print(f"UKF Velocity Magnitude:")
        print(f"  RMSE: {rmse:.4f} m/s")
        print(f"  R²: {r2:.4f}")

        if empirical_model is not None:
            rmse_emp = np.sqrt(mean_squared_error(v_mag_true, results['v_empirical']))
            r2_emp = r2_score(v_mag_true, results['v_empirical'])
            print(f"\nEmpirical Model Direct:")
            print(f"  RMSE: {rmse_emp:.4f} m/s")
            print(f"  R²: {r2_emp:.4f}")

        print("\n" + "=" * 70)
        print("PLANAR WIND ESTIMATION (Should be ~0 for indoor flight)")
        print("=" * 70)
        print(f"Mean wind: [{np.mean(results['wind_x']):.3f}, "
              f"{np.mean(results['wind_y']):.3f}] m/s")
        wind_mag_planar = np.sqrt(results['wind_x'] ** 2 + results['wind_y'] ** 2)
        print(f"Planar wind magnitude: {np.mean(wind_mag_planar):.3f} ± "
              f"{np.std(wind_mag_planar):.3f} m/s")

    return results

def plot_results(results):
    """Plot estimation results."""
    print("\nGenerating plots...")

    time = results['time']
    if time[0] > 1000:
        time = time / 1000.0
    time = time - time[0]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    v_mag_true = np.sqrt(results['vx_true'] ** 2 + results['vy_true'] ** 2 + results['vz_true'] ** 2)
    v_mag_est = np.sqrt(results['vx_est'] ** 2 + results['vy_est'] ** 2 + results['vz_est'] ** 2)
    # Plot 1: Velocity magnitude comparison
    ax = axes[0, 0]
    ax.plot(time, v_mag_true, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(time, v_mag_est, 'r-', linewidth=2, label='UKF Estimate', alpha=0.8)
    ax.plot(time, results['v_empirical'], 'g--', linewidth=1.5, label='Empirical Model', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity Magnitude (m/s)')
    ax.set_title('Velocity Magnitude Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Vx comparison
    ax = axes[0, 1]
    ax.plot(time, results['vx_true'], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(time, results['vx_est'], 'r-', linewidth=2, label='UKF Estimate', alpha=0.8)
    # ax.fill_between(time,
    #                  results['vx_est'] - np.sqrt(results['cov_vx']),
    #                  results['vx_est'] + np.sqrt(results['cov_vx']),
    #                  alpha=0.2, color='red', label='±1σ uncertainty')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Vx (m/s)')
    ax.set_title('X Velocity Component', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Vy comparison
    ax = axes[1, 0]
    ax.plot(time, results['vy_true'], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(time, results['vy_est'], 'r-', linewidth=2, label='UKF Estimate', alpha=0.8)
    # ax.fill_between(time,
    #                  results['vy_est'] - np.sqrt(results['cov_vy']),
    #                  results['vy_est'] + np.sqrt(results['cov_vy']),
    #                  alpha=0.2, color='red', label='±1σ uncertainty')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Vy (m/s)')
    ax.set_title('Y Velocity Component', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Velocity errors
    ax = axes[1, 1]
    vx_error = results['vx_est'] - results['vx_true']
    vy_error = results['vy_est'] - results['vy_true']
    v_mag_error = v_mag_est - v_mag_true

    ax.plot(time, vx_error, 'b-', linewidth=1.5, label='Vx Error', alpha=0.7)
    ax.plot(time, vy_error, 'r-', linewidth=1.5, label='Vy Error', alpha=0.7)
    ax.plot(time, v_mag_error, 'g-', linewidth=1.5, label='|V| Error', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (m/s)')
    ax.set_title('Velocity Estimation Errors', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Wind estimation
    ax = axes[2, 0]
    ax.plot(time, results['wind_x'], 'b-', linewidth=2, label='Wind X')
    ax.plot(time, results['wind_y'], 'r-', linewidth=2, label='Wind Y')
    ax.plot(time, results['wind_mag'], 'g--', linewidth=2, label='Wind Magnitude')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.fill_between(time, -0.2, 0.2, alpha=0.2, color='gray', label='±0.2 m/s band')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Wind Velocity (m/s)')
    ax.set_title('Wind Estimation (should be ~0 indoors)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Magnetic field measurements
    ax = axes[2, 1]
    ax.plot(time, results['Bx'], 'b-', linewidth=1.5, label='Bx', alpha=0.7)
    ax.plot(time, results['By'], 'r-', linewidth=1.5, label='By', alpha=0.7)
    ax.plot(time, results['B_mag'], 'g-', linewidth=1.5, label='B magnitude', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Magnetic Field')
    ax.set_title('Magnetic Field Measurements', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simplified_ukf_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved: simplified_ukf_results.png")

    plt.show()

def main():
    """Main execution."""
    filepath = r"C:\Users\ltjth\Documents\Research\UKF_Data\CF_ARR_Line_EKF0.31.csv"

    print("=" * 70)
    print("WIND ESTIMATION UKF - Q-Based Tuning with Measured R")
    print("=" * 70)

    # Initialize empirical sensor model
    empirical_model = EmpiricalSensorModel(
        a=-0.0001523,
        b=0.03312,
        c=0.1211
    )

    # Load data
    data = load_data(filepath)

    # Calibration
    calibration = None

    # ============================================================
    # MEASUREMENT NOISE (R) - FROM YOUR SENSOR CHARACTERIZATION
    # ============================================================
    # These are NOT tuned - they come from your measurements!

    R_velocity_cov = np.array([[2.83618479e-04, 3.92207351e-05],
                               [3.92207351e-05, 1.04702344e-04]])

    R_magnetic_cov = np.array([[5.0506233, 1.88419894],
                               [1.88419894, 6.63777725]])

    print("\nUsing measured covariance matrices:")
    print(f"Velocity R:\n{R_velocity_cov}")
    print(f"Magnetic R:\n{R_magnetic_cov}")

    # ============================================================
    # PROCESS NOISE (Q) - TUNING PARAMETERS
    # ============================================================
    # These are what you tune for performance!

    TUNING_MODE = 2  # Change this: 1, 2, 3, or 4

    if TUNING_MODE == 1:
        # Conservative: Trust model, smooth estimates
        Q_position = 0.001
        Q_velocity = 0.01
        Q_wind = 0.001
        print("\n>>> TUNING MODE 1: Conservative (trust model)")
        print(f"    Q_vel = {Q_velocity}, Q_wind = {Q_wind}")

    elif TUNING_MODE == 2:
        # Balanced
        Q_position = 0.01
        Q_velocity = 0.1
        Q_wind = 0.01
        print("\n>>> TUNING MODE 2: Balanced")
        print(f"    Q_vel = {Q_velocity}, Q_wind = {Q_wind}")

    elif TUNING_MODE == 3:
        # Aggressive: Trust measurements, fast adaptation
        Q_position = 0.1
        Q_velocity = 1.0
        Q_wind = 0.1
        print("\n>>> TUNING MODE 3: Aggressive (trust measurements)")
        print(f"    Q_vel = {Q_velocity}, Q_wind = {Q_wind}")

    elif TUNING_MODE == 4:
        # Very aggressive: Very fast adaptation
        Q_position = 0.1
        Q_velocity = 10.0
        Q_wind = 1.0
        print("\n>>> TUNING MODE 4: Very aggressive (very fast adaptation)")
        print(f"    Q_vel = {Q_velocity}, Q_wind = {Q_wind}")

    # Run UKF
    results = run_wind_ukf(
        data,
        calibration,
        empirical_model,
        Q_position=Q_position,
        Q_velocity=Q_velocity,
        Q_wind=Q_wind,
        R_velocity_cov=R_velocity_cov,
        R_magnetic_cov=R_magnetic_cov,
        odometry_position_gain=0.6,
        use_odometry_velocity=True
    )

    # Plot results
    plot_results(results)
    plt.show()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nTuning mode used: {TUNING_MODE}")
    print("Change TUNING_MODE to experiment with different Q values")
    print("\nRemember: R matrices are from sensor specs and should NOT be tuned!")



if __name__ == "__main__":
    main()
