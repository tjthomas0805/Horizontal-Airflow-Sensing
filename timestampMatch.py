"""
Module for loading and aligning Crazyflie and Optitrack data
"""
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d


# -----------------------------
# --- Optitrack Functions -----
# -----------------------------
def get_optitrack_start_time(filepath):
    """Extract the start timestamp from Optitrack CSV header"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        first_line = lines[0].strip().split(',')
        timestamp_str = first_line[11].strip()
        return datetime.strptime(timestamp_str, '%Y-%m-%d %I.%M.%S.%f %p')


def load_optitrack_data(filepath):
    """
    Load Optitrack position data from CSV file

    Returns:
        global_times: Unix timestamps (seconds)
        x: X position in meters
        y: Y position in meters
        start_time: datetime object of recording start
    """
    data = np.genfromtxt(filepath, delimiter=',', skip_header=7, filling_values=np.nan)
    start_time = get_optitrack_start_time(filepath)

    elapsed_time = data[:, 1]
    x = data[:, 11] / 1000  # mm -> m
    y = data[:, 12] / 1000

    valid_rows = ~(np.isnan(elapsed_time) | np.isnan(x) | np.isnan(y))
    elapsed_time = elapsed_time[valid_rows]
    x = x[valid_rows]
    y = y[valid_rows]

    global_times = np.array([start_time.timestamp() + t for t in elapsed_time])
    return global_times, x, y, start_time


# -----------------------------
# --- Crazyflie Functions -----
# -----------------------------
def load_crazyflie_data(filepath):
    """
    Load Crazyflie velocity data from CSV file

    Returns:
        timestamps: Unix timestamps (seconds)
        vx: X velocity in m/s
        vy: Y velocity in m/s
    """
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)

    months = data[:, 0].astype(int)
    days = data[:, 1].astype(int)
    hours = data[:, 2].astype(int)
    minutes = data[:, 3].astype(int)
    seconds = data[:, 4].astype(int)
    microseconds = data[:, 5].astype(int)

    timestamps = []
    for i in range(len(months)):
        dt = datetime(2025, months[i], days[i], hours[i], minutes[i],
                      seconds[i], microseconds[i])
        timestamps.append(dt.timestamp())

    timestamps = np.array(timestamps)
    vx = data[:, 6]
    vy = data[:, 7]
    return timestamps, vx, vy


# -----------------------------
# --- Velocity Functions ------
# -----------------------------
def compute_gt_velocity(x, y, time):
    """
    Compute ground truth velocity from position data using finite differences

    Args:
        x: X positions (m)
        y: Y positions (m)
        time: timestamps (seconds)

    Returns:
        time_midpoints: timestamps at velocity midpoints
        vx: X velocity component (m/s)
        vy: Y velocity component (m/s)
        velocity: velocity magnitude (m/s)
    """
    dt = np.diff(time)
    vx = np.diff(x) / dt
    vy = np.diff(y) / dt
    velocity = np.sqrt(vx ** 2 + vy ** 2)
    # Return velocity at midpoints
    time_midpoints = (time[:-1] + time[1:]) / 2
    return time_midpoints, vx, vy, velocity


# -----------------------------
# --- Alignment Functions -----
# -----------------------------
def find_first_crossing(times, signal, threshold):
    """
    Find the first time where signal crosses above threshold

    Args:
        times: timestamp array
        signal: signal values
        threshold: threshold value to cross

    Returns:
        crossing_time: timestamp of first crossing, or None if not found
    """
    above_threshold = signal > threshold

    if not np.any(above_threshold):
        print(f"Warning: Signal never crosses {threshold} threshold")
        return None

    crossing_idx = np.argmax(above_threshold)
    crossing_time = times[crossing_idx]

    return crossing_time


def find_crossing_in_window(times, signal, threshold, start_time, end_time):
    """
    Find the first time where signal crosses above threshold in a time window

    Args:
        times: timestamp array
        signal: signal values
        threshold: threshold value to cross
        start_time: start of search window (seconds relative to times[0])
        end_time: end of search window (seconds relative to times[0])

    Returns:
        crossing_time: timestamp of first crossing in window, or None if not found
    """
    time_rel = times - times[0]

    # Find indices in the search window
    window_mask = (time_rel >= start_time) & (time_rel <= end_time)

    if not np.any(window_mask):
        print(f"Warning: No data in window {start_time} to {end_time} s")
        return None

    window_indices = np.where(window_mask)[0]
    window_times = times[window_indices]
    window_signal = signal[window_indices]

    # Find first crossing above threshold in this window
    above_threshold = window_signal > threshold

    if not np.any(above_threshold):
        print(f"Warning: Signal never crosses {threshold} threshold in window")
        return None

    crossing_idx = window_indices[np.argmax(above_threshold)]
    crossing_time = times[crossing_idx]

    return crossing_time


def align_crazyflie_to_optitrack(optitrack_file, crazyflie_file,
                                 threshold=0.2, opti_window_start=30, opti_window_end=40,
                                 verbose=True):
    """
    Load and align Crazyflie velocity data to Optitrack ground truth

    This function:
    1. Loads Optitrack position data and computes ground truth velocity
    2. Loads Crazyflie velocity data
    3. Finds where CF crosses threshold (anywhere in data)
    4. Finds where Optitrack crosses threshold (in specified window)
    5. Aligns the two datasets by matching these crossing times
    6. Interpolates CF data onto Optitrack timeline

    Args:
        optitrack_file: path to Optitrack CSV file
        crazyflie_file: path to Crazyflie CSV file
        threshold: velocity threshold for alignment (m/s), default 0.2
        opti_window_start: start of Optitrack search window (seconds), default 30
        opti_window_end: end of Optitrack search window (seconds), default 40
        verbose: print alignment information, default True

    Returns:
        Dictionary containing:
            'gt_times': ground truth timestamps (seconds)
            'gt_vx': ground truth X velocity (m/s)
            'gt_vy': ground truth Y velocity (m/s)
            'gt_velocity': ground truth velocity magnitude (m/s)
            'cf_vx_aligned': aligned Crazyflie X velocity (m/s)
            'cf_vy_aligned': aligned Crazyflie Y velocity (m/s)
            'cf_velocity_aligned': aligned Crazyflie velocity magnitude (m/s)
            'time_shift': time shift applied to CF data (seconds)
            'opti_x': Optitrack X positions (m)
            'opti_y': Optitrack Y positions (m)
            'opti_times': Optitrack position timestamps (seconds)
    """
    if verbose:
        print("=" * 60)
        print("Loading data...")
        print("=" * 60)

    # Load data
    opti_times, opti_x, opti_y, opti_start = load_optitrack_data(optitrack_file)
    cf_times, cf_vx, cf_vy = load_crazyflie_data(crazyflie_file)

    if verbose:
        print(f"\nOptitrack samples: {len(opti_times)}")
        print(f"Optitrack duration: {opti_times[-1] - opti_times[0]:.2f} s")
        print(f"\nCrazyflie samples: {len(cf_times)}")
        print(f"Crazyflie duration: {cf_times[-1] - cf_times[0]:.2f} s")

    # Compute ground truth velocity
    gt_times, gt_vx, gt_vy, gt_velocity = compute_gt_velocity(opti_x, opti_y, opti_times)

    # Compute CF speed magnitude
    cf_speed = np.sqrt(cf_vx ** 2 + cf_vy ** 2)

    if verbose:
        print("\n" + "=" * 60)
        print(f"Finding threshold crossings at {threshold} m/s...")
        print("=" * 60)

    # Find CF crossing (anywhere in the data)
    cf_crossing = find_first_crossing(cf_times, cf_speed, threshold)

    # Find Optitrack crossing (in specified window)
    opti_crossing = find_crossing_in_window(gt_times, gt_velocity, threshold,
                                            opti_window_start, opti_window_end)

    if opti_crossing is None or cf_crossing is None:
        raise ValueError("Could not find threshold crossings in data! Check threshold and window parameters.")

    # Calculate time shift needed to align the crossings
    time_shift = opti_crossing - cf_crossing

    if verbose:
        print(f"\nOptitrack crossing at: {opti_crossing - gt_times[0]:.3f} s (relative to start)")
        print(f"Crazyflie crossing at: {cf_crossing - cf_times[0]:.3f} s (relative to start)")
        print(f"Time shift applied to CF: {time_shift:.3f} s")

    # Apply shift to CF data
    cf_times_aligned = cf_times + time_shift

    # Interpolate aligned CF data onto GT timeline
    interp_vx = interp1d(cf_times_aligned, cf_vx, bounds_error=False, fill_value=np.nan)
    interp_vy = interp1d(cf_times_aligned, cf_vy, bounds_error=False, fill_value=np.nan)
    interp_speed = interp1d(cf_times_aligned, cf_speed, bounds_error=False, fill_value=np.nan)

    cf_vx_aligned = interp_vx(gt_times)
    cf_vy_aligned = interp_vy(gt_times)
    cf_velocity_aligned = interp_speed(gt_times)

    # Compute alignment metrics
    valid = ~np.isnan(cf_velocity_aligned) & ~np.isnan(gt_velocity)
    if np.sum(valid) >= 2:
        rmse = np.sqrt(np.mean((cf_velocity_aligned[valid] - gt_velocity[valid]) ** 2))
        correlation = np.corrcoef(cf_velocity_aligned[valid], gt_velocity[valid])[0, 1]

        if verbose:
            print(f"\nAlignment Quality:")
            print(f"  RMSE: {rmse:.4f} m/s")
            print(f"  Correlation: {correlation:.4f}")

    # Return aligned data
    return {
        'gt_times': gt_times,
        'gt_vx': gt_vx,
        'gt_vy': gt_vy,
        'gt_velocity': gt_velocity,
        'cf_vx_aligned': cf_vx_aligned,
        'cf_vy_aligned': cf_vy_aligned,
        'cf_velocity_aligned': cf_velocity_aligned,
        'time_shift': time_shift,
        'opti_x': opti_x,
        'opti_y': opti_y,
        'opti_times': opti_times
    }