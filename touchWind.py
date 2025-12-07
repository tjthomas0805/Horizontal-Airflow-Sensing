"""
Full UKF Implementation from Paper with Direct Magnetic Field Measurements
Estimates: velocity, wind velocity, drag force, interaction force
Based on "Touch the Wind" paper approach

Uses magnetic field components (Bx, By, Bz) directly as measurements
instead of computing tilt angles.

Empirical model: v = -0.0001523*b_mag^2 + 0.03312*b_mag + 0.1211
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import mean_squared_error, r2_score
from timestampMatch import align_crazyflie_to_optitrack
import scipy.signal as signal

from scipy.signal import butter, filtfilt

def lowpass_filter(data, cutoff=5, fs=100, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)

def load_data(filepath):

    df = pd.read_csv(filepath)

    print("\nApplying low-pass filter to magnetic field data (5 Hz cutoff @ 100 Hz)...")
    df['Bx_filt'] = lowpass_filter(df['Bx'].values)
    df['By_filt'] = lowpass_filter(df['By'].values)
    df['Bz_filt'] = lowpass_filter(df['Bz'].values)

    print("\nApplying Crazyflie coordinate transformation...")
    df['Bx_cf'] = df['Bx_filt']
    df['By_cf'] = df['By_filt']
    df['Bz_cf'] = df['Bz_filt']

    df['B_mag'] = np.sqrt(df['Bx_cf']**2 + df['By_cf']**2 + df['Bz_cf']**2)
    df['Bxy_mag'] = np.sqrt(df['Bx_cf']**2 + df['By_cf']**2)

    ...


    # Check for position data
    has_position = all(col in df.columns for col in ['PosX', 'PosY', 'PosZ'])
    # if has_position:
    #     print("Position data found: x, y, z")
    # else:
    #     print("Warning: No position data found")

    # Compute qw from quaternion
    if all(col in df.columns for col in ['qx', 'qy', 'qz']):
        print("Computing qw from quaternion components...")
        qx, qy, qz = df['qx'].values, df['qy'].values, df['qz'].values
        qw_squared = 1.0 - (qx**2 + qy**2 + qz**2)
        qw_squared = np.maximum(qw_squared, 0)
        df['qw'] = np.sqrt(qw_squared)

    if 'Vx' in df.columns:
        V_mag = np.sqrt(df['Vx']**2 + df['Vy']**2 + df['Vz']**2)
        print(f"\nVelocity magnitude: [{V_mag.min():.3f}, {V_mag.max():.3f}] m/s")

    # Print magnetic field statistics
    print(f"\nMagnetic field statistics:")
    print(f"  Bx_cf: [{df['Bx_cf'].min():.2f}, {df['Bx_cf'].max():.2f}]")
    print(f"  By_cf: [{df['By_cf'].min():.2f}, {df['By_cf'].max():.2f}]")
    print(f"  Bz_cf: [{df['Bz_cf'].min():.2f}, {df['Bz_cf'].max():.2f}]")
    print(f"  B_mag: [{df['B_mag'].min():.2f}, {df['B_mag'].max():.2f}]")

    return df


class EmpiricalSensorModel:
    """
    Empirical sensor model based on calibration data.

    Converts magnetic field measurements to velocity using the empirical relationship:
    v = a*b_mag^2 + b*b_mag + c

    This model works directly with magnetic field components (Bx, By, Bz).
    """

    def __init__(self, a=-0.0001523, b=0.03312, c=0.1211):
        """
        Initialize empirical sensor model.

        Parameters:
        -----------
        a, b, c : float
            Coefficients for quadratic model: v = a*b_mag^2 + b*b_mag + c
        """
        self.a = a
        self.b = b
        self.c = c

        print("\n" + "="*70)
        print("EMPIRICAL SENSOR MODEL")
        print("="*70)
        print(f"Model: v = {a}*b_mag² + {b}*b_mag + {c}")

        # Compute sensor sensitivity
        self._analyze_sensitivity()

    def _analyze_sensitivity(self):
        """Analyze sensor sensitivity at different operating points."""
        b_test = np.array([30, 50, 70, 100, 150])
        v_test = self.b_mag_to_velocity(b_test)
        dv_db = 2 * self.a * b_test + self.b

        print(f"\nSensor sensitivity analysis:")
        print(f"  b_mag  |  v (m/s)  |  dv/db")
        print(f"  -------|-----------|--------")
        for i in range(len(b_test)):
            print(f"  {b_test[i]:5.0f}  |  {v_test[i]:7.3f}  |  {dv_db[i]:.5f}")

    def b_mag_to_velocity(self, b_mag):
        """
        Convert magnetic field magnitude to velocity using empirical model.

        Parameters:
        -----------
        b_mag : float or array
            Magnetic field magnitude

        Returns:
        --------
        v : float or array
            Estimated velocity magnitude
        """
        v = self.a * b_mag**2 + self.b * b_mag + self.c
        return v#np.maximum(v, 0)  # Velocity can't be negative

    def velocity_to_b_mag(self, v):
        """
        Convert velocity to expected magnetic field magnitude (inverse model).

        Parameters:
        -----------
        v : float or array
            Velocity magnitude

        Returns:
        --------
        b_mag : float or array
            Expected magnetic field magnitude
        """
        # Solve: v = a*b² + b*b + c for b
        # Using quadratic formula
        discriminant = self.b**2 - 4 * self.a * (self.c - v)
        discriminant = np.maximum(discriminant, 0)

        # Take positive root (since b > 0 and a < 0)
        b_mag = (-self.b + np.sqrt(discriminant)) / (2 * self.a)

        return np.maximum(b_mag, 0)

    def velocity_to_bxy(self, v_mag, v_direction):
        """
        Predict expected Bx, By components given velocity.

        The magnetic field deflection is proportional to drag force,
        which is related to velocity squared times direction.

        Parameters:
        -----------
        v_mag : float
            Velocity magnitude
        v_direction : array (2,) or (3,)
            Unit vector of velocity direction (only xy components used)

        Returns:
        --------
        bx, by : float
            Expected magnetic field components
        """
        # Get expected B_mag from velocity
        b_mag_expected = self.velocity_to_b_mag(v_mag)

        # The deflection direction is opposite to velocity direction
        # (drag pushes sensor opposite to motion)
        if len(v_direction) >= 2:
            vxy = np.array([v_direction[0], v_direction[1]])
            vxy_mag = np.linalg.norm(vxy)

            if vxy_mag > 1e-6:
                # Bxy is proportional to velocity and opposite direction
                # Scale factor relates B_mag to Bxy (approximate)
                bxy_mag = b_mag_expected #* 0.5  # Rough estimate
                bx = bxy_mag * vxy[0] / vxy_mag
                by = bxy_mag * vxy[1] / vxy_mag
            else:
                bx, by = 0.0, 0.0
        else:
            bx, by = 0.0, 0.0

        return bx, by


# def calibrate_magnetic_model(data, empirical_model):
#     """
#     Calibrate the relationship between magnetic field components and velocity.
#
#     This finds the mapping from (Bx, By, Bz) to velocity vector.
#
#     Parameters:
#     -----------
#     data : DataFrame
#         Flight data with magnetic field and velocities
#     empirical_model : EmpiricalSensorModel
#         The empirical sensor model
#
#     Returns:
#     --------
#     calibration : dict
#         Calibration parameters
#     """
#     print("\n" + "="*70)
#     print("CALIBRATING MAGNETIC FIELD TO VELOCITY MAPPING")
#     print("="*70)
#
#     # Filter to moving data only
#     V_mag = np.sqrt(data['Vx']**2 + data['Vy']**2 + data['Vz']**2)
#     moving_mask = V_mag > 0.2
#     data_moving = data[moving_mask].copy()
#
#     print(f"Using {len(data_moving)} moving data points")
#
#     # Get magnetic field components
#     Bx = data_moving['Bx_cf'].values
#     By = data_moving['By_cf'].values
#     Bz = data_moving['Bz_cf'].values
#     B_mag = data_moving['B_mag'].values
#     Bxy_mag = np.sqrt(Bx**2 + By**2)
#
#     # Get velocity components
#     Vx = data_moving['Vx'].values
#     Vy = data_moving['Vy'].values
#     Vz = data_moving['Vz'].values
#     V_mag_data = V_mag[moving_mask].values
#
#     # Compute velocity from empirical model
#     V_empirical = empirical_model.b_mag_to_velocity(B_mag)
#
#     # Find relationship between Bx/By and Vx/Vy
#     # The deflection is caused by drag, which opposes velocity
#     # So Bx should be proportional to -Vx (and similar for By)
#
#     # Linear regression: Bx = k_x * Vx + offset_x
#     # Using least squares
#     valid_mask = V_mag_data > 0.3
#
#     if np.sum(valid_mask) > 50:
#         # Fit Bx vs Vx
#         Vx_valid = Vx[valid_mask]
#         Bx_valid = Bx[valid_mask]
#         k_x = np.sum(Bx_valid * Vx_valid) / np.sum(Vx_valid**2)
#
#         # Fit By vs Vy
#         Vy_valid = Vy[valid_mask]
#         By_valid = By[valid_mask]
#         k_y = np.sum(By_valid * Vy_valid) / np.sum(Vy_valid**2)
#
#         # Fit Bxy_mag vs V_mag
#         Bxy_valid = Bxy_mag[valid_mask]
#         V_valid = V_mag_data[valid_mask]
#         k_mag = np.sum(Bxy_valid * V_valid) / np.sum(V_valid**2)
#
#         print(f"\nLinear calibration coefficients:")
#         print(f"  k_x (Bx/Vx): {k_x:.4f}")
#         print(f"  k_y (By/Vy): {k_y:.4f}")
#         print(f"  k_mag (Bxy/V): {k_mag:.4f}")
#     else:
#         k_x, k_y, k_mag = -10.0, -10.0, 10.0  # Default values
#         print("Warning: Not enough data for calibration, using defaults")
#
#     # Compute correlations
#     corr_x = np.corrcoef(Vx[valid_mask], Bx[valid_mask])[0, 1]
#     corr_y = np.corrcoef(Vy[valid_mask], By[valid_mask])[0, 1]
#     corr_mag = np.corrcoef(V_mag_data[valid_mask], B_mag[valid_mask])[0, 1]
#
#     print(f"\nCorrelations:")
#     print(f"  Vx vs Bx: {corr_x:.4f}")
#     print(f"  Vy vs By: {corr_y:.4f}")
#     print(f"  V_mag vs B_mag: {corr_mag:.4f}")
#
#     # Validate empirical model
#     rmse = np.sqrt(mean_squared_error(V_mag_data, V_empirical))
#     r2 = r2_score(V_mag_data, V_empirical)
#
#     print(f"\nEmpirical model validation:")
#     print(f"  RMSE: {rmse:.4f} m/s")
#     print(f"  R²: {r2:.4f}")
#
#     calibration = {
#         'k_x': k_x,
#         'k_y': k_y,
#         'k_mag': k_mag,
#         'corr_x': corr_x,
#         'corr_y': corr_y,
#     }
#
#     # Create calibration plots
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#
#     # Plot 1: Bx vs Vx
#     ax = axes[0, 0]
#     ax.scatter(Vx[valid_mask], Bx[valid_mask], alpha=0.3, s=5)
#     vx_range = np.linspace(Vx[valid_mask].min(), Vx[valid_mask].max(), 100)
#     ax.plot(vx_range, k_x * vx_range, 'r-', linewidth=2, label=f'k_x = {k_x:.2f}')
#     ax.set_xlabel('Vx (m/s)')
#     ax.set_ylabel('Bx')
#     ax.set_title(f'Bx vs Vx (corr = {corr_x:.3f})')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#
#     # Plot 2: By vs Vy
#     ax = axes[0, 1]
#     ax.scatter(Vy[valid_mask], By[valid_mask], alpha=0.3, s=5)
#     vy_range = np.linspace(Vy[valid_mask].min(), Vy[valid_mask].max(), 100)
#     ax.plot(vy_range, k_y * vy_range, 'r-', linewidth=2, label=f'k_y = {k_y:.2f}')
#     ax.set_xlabel('Vy (m/s)')
#     ax.set_ylabel('By')
#     ax.set_title(f'By vs Vy (corr = {corr_y:.3f})')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#
#     # Plot 3: B_mag vs V_mag
#     ax = axes[1, 0]
#     ax.scatter(V_mag_data, B_mag, alpha=0.3, s=5)
#     ax.set_xlabel('Velocity Magnitude (m/s)')
#     ax.set_ylabel('B_mag')
#     ax.set_title(f'B_mag vs V_mag (corr = {corr_mag:.3f})')
#     ax.grid(True, alpha=0.3)
#
#     # Plot 4: Empirical model validation
#     ax = axes[1, 1]
#     ax.scatter(V_mag_data, V_empirical, alpha=0.3, s=5)
#     ax.plot([0, 3], [0, 3], 'r--', linewidth=2, label='Perfect match')
#     ax.set_xlabel('MoCap Velocity (m/s)')
#     ax.set_ylabel('Empirical Model Velocity (m/s)')
#     ax.set_title(f'Empirical Model Validation (R² = {r2:.3f})')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig('magnetic_calibration.png', dpi=300)
#     print("\nCalibration plot saved: magnetic_calibration.png")
#     plt.show()
#
#     return calibration


class WindEstimationUKF:
    """
    UKF for wind and velocity estimation using direct magnetic field measurements.

    State: [px, py, pz, vx, vy, vz, v_wind_x, v_wind_y]

    Note: Only estimates PLANAR wind (x, y) since single whisker sensor
    can only measure deflection in the horizontal plane.

    Measurements: [Bx, By] - magnetic field components (horizontal only)

    Trust balance (like the paper) is controlled by measurement noise:
    - R_odom (odometry_velocity_noise): higher = trust whisker more
    - R (sensor_noise): higher = trust odometry more
    """

    def __init__(self, dt=0.02, calibration=None, empirical_model=None,
                 odometry_velocity_noise=0.01,
                 odometry_position_gain=0.6,
                 sensor_noise=5.0,
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
        odometry_velocity_noise : float
            Measurement noise for odometry velocity (R_odom).
            HIGHER = trust odometry LESS, trust whisker MORE
            Paper-style tuning parameter.
        odometry_position_gain : float
            How much to trust odometry position (0 = ignore, 1 = fully trust)
        sensor_noise : float
            Measurement noise for Bx, By (R).
            LOWER = trust whisker sensor MORE
        use_odometry_velocity : bool
            If False, completely ignore odometry velocity updates
            (pure whisker-based velocity estimation)
        """
        self.dt = dt
        self.empirical_model = empirical_model

        # Tuning parameters (paper-style: noise covariances)
        self.odometry_velocity_noise = odometry_velocity_noise
        self.odometry_position_gain = odometry_position_gain
        self.use_odometry_velocity = use_odometry_velocity

        # For backwards compatibility
        self.odometry_velocity_gain = 0.5  # Not used in new update, kept for reference

        # Calibration parameters
        if calibration is None:
            calibration = {'k_x': -10.0, 'k_y': -10.0, 'k_mag': 10.0}
        self.k_x = calibration.get('k_x', -10.0)
        self.k_y = calibration.get('k_y', -10.0)
        self.k_mag = calibration.get('k_mag', 10.0)

        # State: [px, py, pz, vx, vy, vz, v_wind_x, v_wind_y]
        # Note: Only 2D wind (no wind_z) - single sensor can only measure planar
        self.dim_x = 8
        self.dim_z = 2  # Measurement: [Bx, By] - horizontal deflection only

        # Drag coefficients (from paper: mu1=0.20, mu2=0.07)
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
        self.ukf.P = np.diag([0.5, 0.5, 0.5,    # Position uncertainty
                             2.0, 2.0, 2.0,     # Velocity uncertainty
                             3.0, 3.0])         # Planar wind uncertainty (x, y only)

        # Process noise
        self.ukf.Q = np.diag([0.1, 0.1, 0.1,    # Position process noise
                             0.1, 0.1, 0.1,     # Velocity process noise
                             0.01, 0.01])       # Planar wind process noise

        # Measurement noise (magnetic field noise) - only Bx, By
        # LOWER = trust sensor MORE
        self.ukf.R = np.diag([sensor_noise, sensor_noise])

        # Store current orientation
        self.current_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.current_omega = np.zeros(3)

        # Baseline Bz (when no deflection) - kept for reference but not used in state
        self.Bz_baseline = 100.0

        print(f"\nUKF Tuning Parameters (Paper-Style):")
        print(f"  Odometry velocity noise (R_odom): {odometry_velocity_noise}")
        print(f"    → Higher = trust whisker MORE")
        print(f"  Whisker sensor noise (R): {sensor_noise}")
        print(f"    → Lower = trust whisker MORE")
        print(f"  Use odometry velocity: {use_odometry_velocity}")

    def state_transition(self, x, dt):
        """State transition: simple kinematics."""
        x_next = x.copy()
        x_next[0:3] = x[0:3] + x[3:6] * dt
        return x_next

    def compute_relative_airflow(self, x):
        """
        Compute relative airflow in body frame.

        Parameters:
        -----------
        x : array (8,)
            State [px, py, pz, vx, vy, vz, wind_x, wind_y]

        Returns:
        --------
        v_inf_body : array (3,)
            Relative airflow in body frame (z component assumes no vertical wind)
        """
        v_world = x[3:6]
        v_wind_world = np.array([x[6], x[7], 0.0])  # Planar wind only (no z component)

        # Convert quaternion to rotation matrix
        quat = self.current_quat
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        R_world_to_body = rot.as_matrix()

        v_inf_world = v_wind_world - v_world  # Same frame!
        v_inf_body = R_world_to_body @ v_inf_world  # Then transform

        return v_inf_body

    def measurement_function(self, x, quat=None, omega=None):
        """
        Measurement function: predict magnetic field components from state.

        Only predicts Bx, By (horizontal deflection) since single sensor
        can only measure planar airflow.

        Parameters:
        -----------
        x : array (8,)
            State [px, py, pz, vx, vy, vz, wind_x, wind_y]

        Returns:
        --------
        z : array (2,)
            Predicted measurement [Bx, By]
        """
        if quat is not None:
            self.current_quat = quat
        if omega is not None:
            self.current_omega = omega

        # Get relative airflow in body frame
        v_inf = self.compute_relative_airflow(x)
        v_world = x[3:6]
        v_wind_world = np.array([x[6], x[7], 0.0])
        # Magnetic field deflection is proportional to airflow
        # Bx ≈ k_x * v_inf_x (linear approximation)
        # By ≈ k_y * v_inf_y

        # Bx_pred = self.k_x * v_inf[0]
        # By_pred = self.k_y * v_inf[1]
        # Predicted Bx, By based on state (in world frame, assuming level flight)
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
        """
        Update with odometry measurements.

        This uses a Kalman-style update where trust is determined by
        the ratio of state covariance (P) to measurement noise (R_odom).

        Like the paper (Section IV-C-2a), odometry is treated as a
        separate measurement update.
        """
        if not self.use_odometry_velocity:
            # Only update position if velocity updates disabled
            innovation_pos = position - self.ukf.x[0:3]
            self.ukf.x[0:3] = self.ukf.x[0:3] + self.odometry_position_gain * innovation_pos
            self.ukf.P[0:3, 0:3] *= 0.9
            return

        # Measurement: [px, py, pz, vx, vy, vz]
        z_odom = np.concatenate([position, velocity_body])

        # Measurement matrix H (which states we're measuring)
        # z = H * x, where x = [px,py,pz, vx,vy,vz, wind_x,wind_y]
        H = np.zeros((6, self.dim_x))
        H[0:3, 0:3] = np.eye(3)  # Position
        H[3:6, 3:6] = np.eye(3)  # Velocity

        # Measurement noise covariance (tuning parameter!)
        # Lower = trust odometry more, Higher = trust whisker more
        R_odom = np.diag([
            0.01, 0.01, 0.01,  # Position noise (usually very accurate)
            self.odometry_velocity_noise,  # Vx noise
            self.odometry_velocity_noise,  # Vy noise
            self.odometry_velocity_noise   # Vz noise
        ])

        # Predicted measurement
        z_pred = H @ self.ukf.x

        # Innovation (measurement residual)
        y = z_odom - z_pred

        # Innovation covariance
        S = H @ self.ukf.P @ H.T + R_odom

        # Kalman gain (this is where the magic happens!)
        # K = P * H^T * S^(-1)
        try:
            K = self.ukf.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback to manual update if matrix inversion fails
            print("manual update")
            self.ukf.x[0:3] += self.odometry_position_gain * (position - self.ukf.x[0:3])
            self.ukf.x[3:6] += self.odometry_velocity_gain * (velocity_body - self.ukf.x[3:6])
            return

        # State update
        self.ukf.x = self.ukf.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.dim_x) - K @ H
        self.ukf.P = I_KH @ self.ukf.P @ I_KH.T + K @ R_odom @ K.T

        # Ensure symmetry and positive definiteness
        self.ukf.P = (self.ukf.P + self.ukf.P.T) / 2
        self.ukf.P = self.ukf.P + np.eye(self.dim_x) * 1e-6

    def update_magnetic(self, Bx, By, Bz, quat, omega):
        """
        Update with magnetic field measurement.

        Only uses Bx, By for update (planar sensing).
        Bz is stored for reference but not used in state estimation.

        Parameters:
        -----------
        Bx, By, Bz : float
            Magnetic field components (only Bx, By used)
        quat : array (4,)
            Current quaternion
        omega : array (3,)
            Current angular velocity
        """
        self.current_quat = quat
        self.current_omega = omega

        # Update Bz baseline (for reference, not used in estimation)
        v_est_mag = np.linalg.norm(self.ukf.x[3:6])
        if v_est_mag < 0.1:
            self.Bz_baseline = 0.95 * self.Bz_baseline + 0.05 * Bz

        # Only use Bx, By for measurement update (planar)
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
        """
        Use empirical model to constrain velocity estimate.

        Parameters:
        -----------
        B_mag : float
            Magnetic field magnitude
        """
        if self.empirical_model is None:
            return

        v_empirical = self.empirical_model.b_mag_to_velocity(B_mag)
        v_est_mag = np.linalg.norm(self.ukf.x[3:6])

        # if v_est_mag > 0.01:
        #     scale_factor = v_empirical / v_est_mag
        #     correction = 0.1 * (scale_factor - 1.0)
        #     correction = np.clip(correction, -0.2, 0.2)
        #     self.ukf.x[3:6] *= (1 + correction)

    def get_state(self):
        """Get current state estimate."""
        # Wind is planar only (x, y), z is assumed 0
        wind_world = np.array([self.ukf.x[6], self.ukf.x[7], 0.0])
        return {
            'position': self.ukf.x[0:3].copy(),
            'velocity_body': self.ukf.x[3:6].copy(),
            'wind_world': wind_world,  # [wind_x, wind_y, 0]
            'v_infinity_body': wind_world - self.ukf.x[3:6].copy(),
            'covariance': self.ukf.P.copy()
        }

    def compute_drag_force(self):
        """Compute drag force from current state (planar wind only)."""
        v_body = self.ukf.x[3:6]
        v_wind_world = np.array([self.ukf.x[6], self.ukf.x[7], 0.0])  # Planar wind

        rot = R.from_quat([self.current_quat[1], self.current_quat[2],
                          self.current_quat[3], self.current_quat[0]])
        R_body_to_world = rot.as_matrix().T
        v_world = R_body_to_world @ v_body

        v_inf = v_wind_world - v_world
        v_inf_norm = np.linalg.norm(v_inf)

        if v_inf_norm < 1e-6:
            return np.zeros(3)

        f_drag_mag = self.mu1 * v_inf_norm + self.mu2 * (v_inf_norm ** 2)
        f_drag_world = f_drag_mag * (v_inf / v_inf_norm)

        return np.zeros(len(f_drag_world))


def run_wind_ukf(data, calibration, empirical_model=None,
                 odometry_velocity_noise=0.01,
                 odometry_position_gain=0.6,
                 sensor_noise=5.0,
                 use_odometry_velocity=True):
    """
    Run the wind estimation UKF with direct magnetic field measurements.

    Paper-Style Tuning Guide:
    -------------------------
    The Kalman gain K automatically balances trust based on noise covariances:

        K = P * H^T * (H*P*H^T + R)^(-1)

    To trust the WHISKER SENSOR more:
        - Increase odometry_velocity_noise (e.g., 2.0)
        - Decrease sensor_noise (e.g., 0.5)

    To trust ODOMETRY more:
        - Decrease odometry_velocity_noise (e.g., 0.01)
        - Increase sensor_noise (e.g., 5.0)
    """
    print("\n" + "="*70)
    print("RUNNING WIND ESTIMATION UKF (Direct Magnetic Field)")
    print("="*70)

    # Determine time step
    if 'time' in data.columns:
        time = data['time'].values
        dt = np.median(np.diff(time))
    else:
        dt = 0.02
        time = np.arange(len(data)) * dt

    if time[0] > 1000:
        time = time / 1000.0

    print(f"Time step: {dt:.4f} s ({1/dt:.1f} Hz)")

    # Initialize filter with tuning parameters
    wind_ukf = WindEstimationUKF(
        dt=dt,
        calibration=calibration,
        empirical_model=empirical_model,
        odometry_velocity_noise=odometry_velocity_noise,
        odometry_position_gain=odometry_position_gain,
        sensor_noise=sensor_noise,
        use_odometry_velocity=use_odometry_velocity
    )

    # Initialize Bz baseline from first few samples
    wind_ukf.Bz_baseline = data['Bz_cf'].iloc[:100].mean()
    print(f"Bz baseline: {wind_ukf.Bz_baseline:.2f}")

    # Initialize with first position and velocity
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

        # Prediction step
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

        # Update with magnetic field measurement
        Bx = data['Bx_cf'].iloc[i]
        By = data['By_cf'].iloc[i]
        Bz = data['Bz_cf'].iloc[i]
        wind_ukf.update_magnetic(Bx, By, Bz, quat, omega)

        # Update using empirical model
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

        f_drag = wind_ukf.compute_drag_force()
        results['drag_x'][i] = f_drag[0]
        results['drag_y'][i] = f_drag[1]
        results['drag_z'][i] = f_drag[2]

        # if i % 100 == 0:
        #     print(f"  Progress: {i}/{len(data)} ({100*i/len(data):.1f}%)")

    print("UKF complete!")

    # Print performance
    if 'Vx' in data.columns:
        print("\n" + "="*70)
        print("VELOCITY ESTIMATION PERFORMANCE")
        print("="*70)

        v_mag_true = np.sqrt(results['vx_true']**2 + results['vy_true']**2 + results['vz_true']**2)
        v_mag_est = np.sqrt(results['vx_est']**2 + results['vy_est']**2 + results['vz_est']**2)

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

        print("\n" + "="*70)
        print("PLANAR WIND ESTIMATION (Should be ~0 for indoor flight)")
        print("="*70)
        print(f"Mean wind: [{np.mean(results['wind_x']):.3f}, "
              f"{np.mean(results['wind_y']):.3f}] m/s (Z not estimated)")
        wind_mag_planar = np.sqrt(results['wind_x']**2 + results['wind_y']**2)
        print(f"Planar wind magnitude: {np.mean(wind_mag_planar):.3f} ± "
              f"{np.std(wind_mag_planar):.3f} m/s")

    return results


def plot_results(results, save_path='wind_ukf_magnetic_results.png'):
    """Plot results with improved readability for overlapping signals."""
    print("\nGenerating plots...")

    time = results['time']
    if time[0] > 1000:
        time = time / 1000.0
    time = time - time[0]

    # Compute derived quantities
    v_mag_true = np.sqrt(results['vx_true']**2 + results['vy_true']**2 + results['vz_true']**2)
    v_mag_est = np.sqrt(results['vx_est']**2 + results['vy_est']**2 + results['vz_est']**2)
    drag_mag = np.sqrt(results['drag_x']**2 + results['drag_y']**2 + results['drag_z']**2)

    # Compute errors
    vx_error = results['vx_est'] - results['vx_true']
    vy_error = results['vy_est'] - results['vy_true']
    vz_error = results['vz_est'] - results['vz_true']
    v_mag_error = v_mag_est - v_mag_true


    # ========== FIGURE 1: Main Results ==========
    fig1, axes = plt.subplots(2, 1)

    # Plot 1a: Vx comparison with offset view
    ax = axes[0]
    ax.plot(time, v_mag_true, 'b-', linewidth=2, label='CF Velocity Estimate', alpha=0.8)
    ax.plot(time, v_mag_est, 'r-', linewidth=2, label='CF+Flow Sensor Fused Estimate', alpha=0.8)
    if np.any(results['v_empirical'] > 0):
        ax.plot(time, results['v_empirical'], 'g-', linewidth=2, label='Flow Sensor', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity Magnitude (m/s)')
    ax.set_title('Velocity Magnitude Comparison', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    #ax.set_xlim([0, 8])
    #ax.set_xlim([0, 6])
    wind_mag_planar = np.sqrt(results['wind_x'] ** 2 + results['wind_y'] ** 2)
    # offset = 0  # Small offset to see both lines
    # ax.plot(time, results['vx_true'], 'b-', linewidth=2, label='Vx EKF Estimate', alpha=0.8)
    # ax.plot(time, results['vx_est'] + offset, 'r-', linewidth=1.5, label=f'Vx UKF Est)', alpha=0.8)
    # ax.fill_between(time, results['vx_true'], results['vx_est'], alpha=0.3, color='gray', label='Difference')
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Velocity (m/s)')
    # ax.set_title('Vx: Ground Truth vs UKF Estimate', fontweight='bold')
    # ax.legend(loc='upper right')
    # ax.grid(True, alpha=0.3)
    # ax.set_xlim([0, min(120, time[-1])])

    # Plot 1b: Vy comparison with offset view
    # ax = axes[1]
    # ax.plot(time, results['vy_true'], 'b-', linewidth=2, label='Vy EKF Estimate', alpha=0.8)
    # ax.plot(time, results['vy_est'] + offset, 'r-', linewidth=1.5, label=f'Vy UKF Est ', alpha=0.8)
    # ax.fill_between(time, results['vy_true'], results['vy_est'], alpha=0.3, color='gray', label='Difference')
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Velocity (m/s)')
    # ax.set_title('Vy: Ground Truth vs UKF Estimate', fontweight='bold')
    # ax.legend(loc='upper right')
    # ax.grid(True, alpha=0.3)
    # ax.set_xlim([0, min(120, time[-1])])

    # Plot 2a: Velocity errors over time
    ax = axes[1]
    ax.plot(time, results['B_mag'], 'b-', linewidth=1.5, alpha=0.7)
    # ax.plot(time, vy_error, 'r-', linewidth=1.5, label='Vy Error', alpha=0.7)
    # ax.plot(time, vz_error, 'g-', linewidth=1.5, label='Vz Error', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    #ax.fill_between(time, -0.1, 0.1, alpha=0.2, color='green', label='±0.1 m/s band')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (m/s)')
    ax.set_title('Velocity Estimation Error (Fusion - CF EKF)', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    #ax.set_xlim([0, 8])
    #ax.set_ylim([-0.5, 0.5])

    # # # Plot 2b: Velocity magnitude with error band
    # ax = axes[2]
    # ax.plot(time, results['wind_mag'], 'b-', linewidth=2, label='Wind', alpha=0.8)
    # ax.plot(time, results['B_mag'], 'b-', linewidth=2, label='Raw B Signal', alpha=0.8)
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Velocity Magnitude (m/s)')
    # ax.set_title('Velocity Magnitude Comparison', fontweight='bold')
    # ax.legend(loc='upper right')
    # ax.grid(True, alpha=0.3)
    # ax.set_xlim([0, min(120, time[-1])])
    # wind_mag_planar = np.sqrt(results['wind_x'] ** 2 + results['wind_y'] ** 2)
    # # Plot 3a: Wind estimation (planar only)
    # ax = axes[2, 0]
    # ax.plot(time, wind_mag_planar, 'b--', linewidth=2, label='Planar Wind Magnitude')
    # # ax.plot(time, results['wind_x'], 'b-', linewidth=2, label='Wind X')
    # # ax.plot(time, results['wind_y'], 'r-', linewidth=2, label='Wind Y')
    # # Note: wind_z is always 0 for single planar sensor
    # ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    # ax.fill_between(time, -0.2, 0.2, alpha=0.2, color='gray', label='±0.2 m/s band')
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Wind Velocity (m/s)')
    # ax.set_title('Planar Wind Estimation (X-Y only, should be ~0 indoors)', fontweight='bold')
    # ax.legend(loc='upper right')
    # ax.grid(True, alpha=0.3)
    # ax.set_xlim([0, min(120, time[-1])])
    # ax.set_ylim([-1, 1])
    #
    # # Plot 3b: Drag force magnitude
    # ax = axes[2, 1]
    # #ax.quiver(results['px_est'], results['py_est'], results['wind_x'], results['wind_y'])#, [C], /, ** kwargs)
    # ax.quiver(results['px_est'], results['py_est'], abs(results['vx_est']-results['vx_true']), abs(results['vy_est']-results['vy_true']))
    # #ax.plot(time, drag_mag, 'r-', linewidth=2, label='Drag Force')
    # # Wind magnitude is planar (sqrt of wind_x^2 + wind_y^2)
    # wind_mag_planar = np.sqrt(results['wind_x']**2 + results['wind_y']**2)
    # #ax.plot(time, wind_mag_planar-v_mag_est, 'b--', linewidth=2, label='Planar Wind Magnitude')
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Magnitude')
    # ax.set_title('Drag Force & Planar Wind Magnitude', fontweight='bold')
    # ax.legend(loc='upper right')
    # ax.grid(True, alpha=0.3)
   # ax.set_xlim([0, min(120, time[-1])])

    plt.tight_layout()
    #plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {save_path}")
    plt.show()

   #  # ========== FIGURE 2: Detailed Analysis ==========
   #  fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
   #
   #  # Plot: Scatter plot - True vs Estimated
   #  ax = axes2[0, 0]
   #  ax.scatter(results['vx_true'], results['vx_est'], alpha=0.3, s=5, c='blue', label='Vx')
   #  ax.scatter(results['vy_true'], results['vy_est'], alpha=0.3, s=5, c='red', label='Vy')
   #  lims = [-2, 2]
   #  ax.plot(lims, lims, 'k--', linewidth=2, label='Perfect match')
   #  ax.set_xlabel('Ground Truth Velocity (m/s)')
   #  ax.set_ylabel('UKF Estimated Velocity (m/s)')
   #  ax.set_title('Scatter: Ground Truth vs Estimate', fontweight='bold')
   #  ax.legend()
   #  ax.grid(True, alpha=0.3)
   #  ax.set_xlim(lims)
   #  ax.set_ylim(lims)
   #  ax.set_aspect('equal')
   #
   #  # Plot: Error histogram
   #  ax = axes2[0, 1]
   #  all_errors = np.concatenate([vx_error, vy_error, vz_error])
   #  ax.hist(all_errors, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
   #  ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
   #  ax.axvline(x=np.mean(all_errors), color='g', linestyle='-', linewidth=2, label=f'Mean: {np.mean(all_errors):.4f}')
   #  ax.axvline(x=np.std(all_errors), color='orange', linestyle=':', linewidth=2, label=f'Std: {np.std(all_errors):.4f}')
   #  ax.axvline(x=-np.std(all_errors), color='orange', linestyle=':', linewidth=2)
   #  ax.set_xlabel('Velocity Error (m/s)')
   #  ax.set_ylabel('Density')
   #  ax.set_title('Error Distribution (All Components)', fontweight='bold')
   #  ax.legend()
   #  ax.grid(True, alpha=0.3)
   #
   #  # Plot: Zoomed window (first 20 seconds or subset)
   #  ax = axes2[1, 0]
   #  zoom_end = min(20, time[-1])
   #  zoom_mask = time <= zoom_end
   #  ax.plot(time[zoom_mask], results['vx_true'][zoom_mask], 'b-', linewidth=2.5, label='Vx Truth')
   #  ax.plot(time[zoom_mask], results['vx_est'][zoom_mask], 'r--', linewidth=2, label='Vx Est')
   #  ax.plot(time[zoom_mask], results['vy_true'][zoom_mask], 'g-', linewidth=2.5, label='Vy Truth')
   #  ax.plot(time[zoom_mask], results['vy_est'][zoom_mask], 'm--', linewidth=2, label='Vy Est')
   #  ax.set_xlabel('Time (s)')
   #  ax.set_ylabel('Velocity (m/s)')
   #  ax.set_title(f'Zoomed View: First {zoom_end:.0f} Seconds', fontweight='bold')
   #  ax.legend(loc='upper right')
   #  ax.grid(True, alpha=0.3)
   #
   #  # Plot: Magnetic field vs Velocity
   #  ax = axes2[1, 1]
   #  ax.scatter(results['vx_true'], results['Bx'], alpha=0.3, s=5, c='blue', label='Bx vs Vx')
   #  ax.scatter(results['vy_true'], results['By'], alpha=0.3, s=5, c='red', label='By vs Vy')
   #  ax.set_xlabel('Velocity (m/s)')
   #  ax.set_ylabel('Magnetic Field')
   #  ax.set_title('Magnetic Field vs Velocity (Calibration Check)', fontweight='bold')
   #  ax.legend()
   #  ax.grid(True, alpha=0.3)
   #
   #  plt.tight_layout()
   #  detailed_save_path = save_path.replace('.png', '_detailed.png')
   #  plt.savefig(detailed_save_path, dpi=300, bbox_inches='tight')
   #  print(f"Detailed plot saved: {detailed_save_path}")
   #  plt.show()
   #
   #  # ========== FIGURE 3: Statistics Summary ==========
   #  fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
   #
   #  # Plot: RMSE by component
   #  ax = axes3[0]
   #  rmse_vx = np.sqrt(np.mean(vx_error**2))
   #  rmse_vy = np.sqrt(np.mean(vy_error**2))
   #  rmse_vz = np.sqrt(np.mean(vz_error**2))
   #  rmse_mag = np.sqrt(np.mean(v_mag_error**2))
   #
   #  components = ['Vx', 'Vy', 'Vz', '|V|']
   #  rmse_values = [rmse_vx, rmse_vy, rmse_vz, rmse_mag]
   #  colors = ['steelblue', 'coral', 'seagreen', 'purple']
   #
   #  bars = ax.bar(components, rmse_values, color=colors, edgecolor='black', alpha=0.8)
   #  ax.set_ylabel('RMSE (m/s)')
   #  ax.set_title('RMSE by Velocity Component', fontweight='bold')
   # # ax.grid(True, alpha=0.3, axis='PosY')
   #
   #  # Add value labels on bars
   #  for bar, val in zip(bars, rmse_values):
   #      ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
   #              f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
   #
   #  # Plot: Running RMSE over time
   #  ax = axes3[1]
   #  window = 100  # samples
   #  running_rmse = []
   #  running_time = []
   #  for i in range(window, len(time), window//2):
   #      window_error = v_mag_error[i-window:i]
   #      running_rmse.append(np.sqrt(np.mean(window_error**2)))
   #      running_time.append(time[i])
   #
   #  ax.plot(running_time, running_rmse, 'b-', linewidth=2)
   #  ax.axhline(y=rmse_mag, color='r', linestyle='--', linewidth=2, label=f'Overall RMSE: {rmse_mag:.4f}')
   #  ax.set_xlabel('Time (s)')
   #  ax.set_ylabel('Running RMSE (m/s)')
   #  ax.set_title(f'Running RMSE (window={window} samples)', fontweight='bold')
   #  ax.legend()
   #  ax.grid(True, alpha=0.3)
   #  ax.set_xlim([0, min(120, time[-1])])
   #
   #  plt.tight_layout()
   #  stats_save_path = save_path.replace('.png', '_stats.png')
   #  plt.savefig(stats_save_path, dpi=300, bbox_inches='tight')
   #  print(f"Statistics plot saved: {stats_save_path}")
   #  plt.show()

    # Print summary statistics
    print("\n" + "="*70)
    print("ESTIMATION STATISTICS SUMMARY")
    print("="*70)
    print(f"RMSE Vx: {rmse_vx:.4f} m/s")
    print(f"RMSE Vy: {rmse_vy:.4f} m/s")
    print(f"RMSE Vz: {rmse_vz:.4f} m/s")
    print(f"RMSE |V|: {rmse_mag:.4f} m/s")
    print(f"\nMean Error: {np.mean(all_errors):.4f} m/s")
    print(f"Std Error:  {np.std(all_errors):.4f} m/s")
    print(f"Max Error:  {np.max(np.abs(all_errors)):.4f} m/s")
    print(f"\nPlanar Wind Estimate Mean: [{np.mean(results['wind_x']):.4f}, {np.mean(results['wind_y']):.4f}] m/s (Z assumed 0)")
    print(f"Planar Wind Estimate Std:  [{np.std(results['wind_x']):.4f}, {np.std(results['wind_y']):.4f}] m/s")


def main():
    """Main execution."""
    # UPDATE THIS PATH to your data file
    filepath = r"C:\Users\ltjth\Documents\Research\UKF_Data\CF_ARR_Square_EKF0.5ms_FAN1.csv"
    optitrack = r"C:\Users\ltjth\Documents\Research\UKF_Data\Optitrack\OPT_ARR_Square_EKF0.5ms_FAN.csv"
    #filepath = r"C:\Users\ltjth\Documents\Research\UKF_Data\soft_square_wind0.51.csv"
    #filepath = r"C:\Users\ltjth\Documents\Research\UKF_Data\fan_flyby1.csv"
    print("="*70)
    print("WIND ESTIMATION UKF - Direct Magnetic Field Measurements")
    print("="*70)

    # Initialize empirical sensor model
    empirical_model = EmpiricalSensorModel(
        a=-0.0001523,
        b=0.03312,
        c=0.1211
    )

    # Load data
    data = load_data(filepath)

    # Calibrate magnetic field to velocity mapping
    calibration = None#calibrate_magnetic_model(data, empirical_model)

    # ============================================================
    # TUNING PARAMETERS (Paper-Style)
    # ============================================================
    # The paper balances trust via measurement noise covariances:
    #
    #   K = P * H^T * (H*P*H^T + R)^(-1)
    #
    # - Large R (measurement noise) → small K → trust measurement LESS
    # - Small R → large K → trust measurement MORE
    #
    # Two R matrices to tune:
    #   1. R_odom (odometry_velocity_noise): noise on velocity from odometry
    #   2. R (sensor_noise): noise on Bx, By from whisker
    #
    # ============================================================
    #
    # Option 1: Trust odometry more (like having good motion capture)
    #   odometry_velocity_noise = 0.01  (low noise = trust it)
    #   sensor_noise = 5.0              (high noise = don't trust as much)
    #
    # Option 2: Balanced trust
    #   odometry_velocity_noise = 0.5
    #   sensor_noise = 2.0
    #
    # Option 3: Trust whisker sensor more
    #   odometry_velocity_noise = 2.0   (high noise = don't trust)
    #   sensor_noise = 0.5              (low noise = trust it)
    #
    # Option 4: Pure whisker estimation (no odometry velocity)
    #   use_odometry_velocity = False
    #   sensor_noise = 1.0
    #
    # ============================================================

    # Choose your tuning here:
    TUNING_MODE = 2  # Change this: 1, 2, 3, or 4

    if TUNING_MODE == 1:
        # Trust odometry more
        odometry_velocity_noise = 0.01
        sensor_noise = 5.0
        use_odometry_velocity = True
        print("\n>>> TUNING MODE 1: Trust odometry more")
        print("    R_odom = 0.01 (low), R_sensor = 5.0 (high)")
    elif TUNING_MODE == 2:
        # Balanced trust
        odometry_velocity_noise = 0.5
        sensor_noise = 2.0
        use_odometry_velocity = True
        print("\n>>> TUNING MODE 2: Balanced trust")
        print("    R_odom = 0.5, R_sensor = 2.0")
    elif TUNING_MODE == 3:
        # Trust whisker sensor more
        odometry_velocity_noise = 2.0
        sensor_noise = 0.5
        use_odometry_velocity = True
        print("\n>>> TUNING MODE 3: Trust whisker sensor more")
        print("    R_odom = 2.0 (high), R_sensor = 0.5 (low)")
    elif TUNING_MODE == 4:
        # Pure whisker estimation
        odometry_velocity_noise = 100.0  # Effectively ignored
        sensor_noise = 1.0
        use_odometry_velocity = False
        print("\n>>> TUNING MODE 4: Pure whisker estimation (no odometry velocity)")
        print("    R_sensor = 1.0")

    # Run UKF with tuning parameters
    results = run_wind_ukf(
        data,
        calibration,
        empirical_model,
        odometry_velocity_noise=odometry_velocity_noise,
        odometry_position_gain=0.6,  # Keep position from odometry
        sensor_noise=sensor_noise,
        use_odometry_velocity=use_odometry_velocity
    )

    # Plot results
    plot_results(results, save_path='wind_ukf_magnetic_results.png')
    #plt.show()
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nKey outputs:")
    print("  - magnetic_calibration.png: Shows Bx/By vs Vx/Vy relationships")
    print("  - wind_ukf_magnetic_results.png: Full UKF results")
    print(f"\nTuning mode used: {TUNING_MODE}")
    print("  Change TUNING_MODE in main() to try different sensor/odometry balance")


if __name__ == "__main__":
    main()