import logging
import time
import os
import csv
from datetime import datetime
from threading import Thread
import keyboard
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.platformservice import PlatformService
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.log import LogConfig
import numpy as np

# === CONFIGURATION ===
URI = 'radio://0/80/2M/E7E7E7E7E7'  # <-- change if needed
directory_path = r"C:\Users\ltjth\Documents\Research\VelocityLogs"
base_filename = "surgeTest"
file_extension = ".csv"

# Speeds for square flights (m/s)
SPEEDS = [0.5]
SIDE_DISTANCE = 3.0  # meters
HEIGHT = 0.8  # takeoff height

# PID Controller Variables
Kp = 0.6
Ki = 0
Kd = 0.08
integral = 0
previous_error = 0
previous_time = 0
current_time = 0

# Drone Variables
drone_heading = 0

# Navigation Vars
min_flow_threshold = 8
angle_threshold = 10
desired_flow_angle = 180
flowMap = []
#once you detect flow add these as a tuple to flowMap

# Wind source navigation parameters
SEARCH_SPEED = 0.3
APPROACH_SPEED = 1
MAX_FLOW_THRESHOLD = 70  # When to consider we've reached the source
TURN_RATE = 45  # degrees/second for turning towards source


# === FILE HANDLING ===
def get_log_filename():
    file_number = 1
    while True:
        csv_filename = f"{base_filename}{file_number}{file_extension}"
        full_path = os.path.join(directory_path, csv_filename)
        if not os.path.exists(full_path):
            return full_path
        file_number += 1


# === HELPER FUNCTIONS ===

def navigate_to_wind_source(mc, logger):
    """
    Main navigation function to find and approach wind source
    """
    print("Starting wind source navigation...")

    # Phase 1: Search for wind
    search_for_wind(mc, logger)

    # Phase 2: Navigate towards source
    approach_wind_source(mc, logger)


def search_for_wind(mc, logger):
    """
    Search phase - move forward slowly until wind is detected
    """
    print("Searching for wind...")

    while logger.flowMag < min_flow_threshold:
        print(f"No wind detected (mag: {logger.flowMag:.2f}), continuing search...")
        mc.start_forward(SEARCH_SPEED)
        time.sleep(0.1)

    print(f"Wind detected! Magnitude: {logger.flowMag:.2f}")
    mc.stop()
    time.sleep(0.2)


def approach_wind_source(mc, logger):
    """
    Approach phase - navigate towards wind source using flow direction
    """
    print("Approaching wind source...")

    while logger.flowMag < MAX_FLOW_THRESHOLD:
        if logger.flowMag < min_flow_threshold:
            print("Lost wind signal, resuming search...")
            search_for_wind(mc, logger)
            continue

        # Calculate wind source direction (opposite to flow direction)
        source_angle = (logger.flowAngle + 180) % 360
        flowMap.append((logger.droneX, logger.droneY, logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))
        # Convert to velocity components (NED frame)
        # 0° = North, 90° = East, 180° = South, 270° = West
        vx = APPROACH_SPEED * np.cos(np.radians(source_angle))  # North component
        vy = APPROACH_SPEED * np.sin(np.radians(source_angle))  # East component

        print(f"Flow: {logger.flowMag:.2f} @ {logger.flowAngle:.1f}° | "
              f"Source: {source_angle:.1f}° | "
              f"Velocity: vx={vx:.2f}, vy={vy:.2f}")

        # Move towards source
        mc.start_linear_motion(vx, vy, 0)
        time.sleep(0.1)

    print(f"Reached wind source! Flow magnitude: {logger.flowMag:.2f}")
    mc.stop()
    time.sleep(1)

    # Land at source
    print("Landing at wind source...")
    mc.start_down(0.3)
    time.sleep(2)
    mc.land()


def alternative_turn_and_surge(mc, logger):
    """
    Alternative approach using turning and surging
    """
    print("Using turn-and-surge approach...")

    while logger.flowMag < MAX_FLOW_THRESHOLD:
        if logger.flowMag < min_flow_threshold:
            print("No wind detected, searching...")
            mc.start_forward(SEARCH_SPEED)
            time.sleep(0.1)
            continue

        # Calculate error - we want to face opposite to flow direction
        target_heading = (logger.flowAngle + 180) % 360
        current_heading = 0  # We don't have heading sensor, so assume 0 = North

        # Calculate shortest angular distance
        error = target_heading - current_heading
        if error > 180:
            error -= 360
        elif error < -180:
            error += 360

        print(f"Flow: {logger.flowMag:.2f} @ {logger.flowAngle:.1f}° | "
              f"Target heading: {target_heading:.1f}° | Error: {error:.1f}°")

        # If aligned with source, surge forward
        if abs(error) < angle_threshold:
            print("Aligned with source, surging forward...")
            mc.start_forward(APPROACH_SPEED)
        else:
            # Turn towards source
            turn_speed = min(abs(error) / 180 * TURN_RATE, TURN_RATE)
            if error > 0:
                print(f"Turning left at {turn_speed:.1f}°/s")
                mc.start_turn_left(turn_speed)
            else:
                print(f"Turning right at {turn_speed:.1f}°/s")
                mc.start_turn_right(turn_speed)

        time.sleep(0.1)

    print(f"Reached wind source! Flow magnitude: {logger.flowMag:.2f}")
    mc.stop()
    mc.start_down(0.3)
    mc.land()


# === LOGGING THREAD ===
class LoggerThread(Thread):
    def __init__(self, cf, writer):
        super().__init__()
        self.cf = cf
        self.writer = writer
        self.running = True

        # Moving average filter for flow angle
        self.angle_window = []
        self.angle_window_size = 5  # Reduced for faster response

        # Calibration parameters
        self.calib_window = 100
        self.bx_window = []
        self.by_window = []
        self.bx_offset = 0
        self.by_offset = 0
        self.calibrated = False

        # Time variables
        self.month = 0
        self.day = 0
        self.hour = 0
        self.minute = 0
        self.second = 0
        self.microsecond = 0

        # Drone state variables
        self.droneX = 0
        self.droneY = 0
        self.droneVX = 0
        self.droneVY = 0

        # Flow variables
        self.latest_bx = 0.0
        self.latest_by = 0.0
        self.flowAngle = 0.0
        self.flowMag = 0.0

    def moving_average_angle(self, new_angle):
        """Apply moving average filter to angle measurements"""
        self.angle_window.append(new_angle)
        if len(self.angle_window) > self.angle_window_size:
            self.angle_window.pop(0)

        # Convert to unit vectors and average
        sin_sum = sum(np.sin(np.radians(a)) for a in self.angle_window)
        cos_sum = sum(np.cos(np.radians(a)) for a in self.angle_window)

        avg_angle = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
        return avg_angle

    def run(self):
        log_conf = LogConfig(name='Logger', period_in_ms=50)  # 20 Hz
        log_conf.add_variable('stateEstimate.vx', 'float')
        log_conf.add_variable('stateEstimate.vy', 'float')
        log_conf.add_variable('stateEstimate.x', 'float')
        log_conf.add_variable('stateEstimate.y', 'float')
        log_conf.add_variable('stateEstimate.z', 'float')
        log_conf.add_variable('windSensor.flowX', 'int16_t')
        log_conf.add_variable('windSensor.flowY', 'int16_t')

        def log_data(timestamp, data, logconf):
            bx = data['windSensor.flowX']
            by = data['windSensor.flowY']

            # Calibration step
            if not self.calibrated:
                if len(self.bx_window) < self.calib_window:
                    self.bx_window.append(bx)
                    self.by_window.append(by)
                    return
                else:
                    self.bx_offset = sum(self.bx_window) / len(self.bx_window)
                    self.by_offset = sum(self.by_window) / len(self.by_window)
                    self.calibrated = True
                    print(f"Calibration done: Bx offset={self.bx_offset:.2f}, By offset={self.by_offset:.2f}")

            # Update time
            now = datetime.now()
            self.month = now.month
            self.day = now.day
            self.hour = now.hour
            self.minute = now.minute
            self.second = now.second
            self.microsecond = now.microsecond

            # Update drone state
            self.droneX = data['stateEstimate.x']
            self.droneY = data['stateEstimate.y']
            self.droneVX = data['stateEstimate.vx']
            self.droneVY = data['stateEstimate.vy']

            # Calculate calibrated flow values
            bx_cal = bx - self.bx_offset
            by_cal = by - self.by_offset

            # Update flow measurements
            self.latest_bx = bx_cal
            self.latest_by = by_cal

            # Calculate flow angle and magnitude
            raw_angle = (np.arctan2(by_cal, bx_cal) * (180 / np.pi))-180 % 360
            self.flowAngle = self.moving_average_angle(raw_angle)
            self.flowMag = np.sqrt(bx_cal ** 2 + by_cal ** 2)

            # Log data to CSV
            if self.running:
                self.writer.writerow({
                    "Month": now.month,
                    "Day": now.day,
                    "Hour": now.hour,
                    "Minute": now.minute,
                    "Second": now.second,
                    "Microsecond": now.microsecond,
                    "Speed": np.sqrt(self.droneVX ** 2 + self.droneVY ** 2),
                    "Vx": data['stateEstimate.vx'],
                    "Vy": data['stateEstimate.vy'],
                    "PosX": data['stateEstimate.x'],
                    "PosY": data['stateEstimate.y'],
                    "PosZ": data['stateEstimate.z'],
                    "Bx": bx_cal,
                    "By": by_cal,
                    "Flow Mag": self.flowMag,
                    "Flow Angle": self.flowAngle
                })

        def log_error(logconf, msg):
            print(f"Logging error: {msg}")

        # Start logging
        self.cf.log.add_config(log_conf)
        log_conf.data_received_cb.add_callback(log_data)
        log_conf.error_cb.add_callback(log_error)

        try:
            log_conf.start()
            while self.running:
                time.sleep(0.1)
            log_conf.stop()
        except Exception as e:
            print("Logging error:", e)


# === MAIN PROGRAM ===
if __name__ == '__main__':
    cflib.crtp.init_drivers(enable_debug_driver=False)
    log_path = get_log_filename()

    with open(log_path, mode="w", newline='') as csv_file:
        fieldnames = [
            "Month", "Day", "Hour", "Minute", "Second", "Microsecond",
            "Speed", "Vx", "Vy", "PosX", "PosY", "PosZ", "Bx", "By",
            "Flow Mag", "Flow Angle"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            cf = scf.cf
            platform = PlatformService(crazyflie=cf)

            print("Connected — sending ARM request…")
            platform.send_arming_request(True)
            print("Armed! (but not taking off)")

            # Start logging thread
            logger = LoggerThread(cf, writer)
            logger.start()

            # Initialize motion commander
            mc = MotionCommander(scf, default_height=HEIGHT)


            # Killswitch handler
            def killswitch():
                print("\nESC pressed! Landing immediately...")
                #mc.stop()
                mc.start_down(0.3)
                mc.land()
                time.sleep(5)
                platform.send_arming_request(False)
                logger.running = False
                print("Disarmed. Exiting program.")
                os._exit(0)


            # Take off
            print("Taking off...")
            mc.take_off(HEIGHT)
            time.sleep(2)

            # Set up killswitch
            keyboard.add_hotkey('esc', killswitch)

            try:
                # Wait for calibration
                while not logger.calibrated:
                    print("Waiting for calibration...")
                    time.sleep(1)

                print("Calibrated! Starting wind source navigation...")
                time.sleep(2)

                # Choose navigation method:
                # Method 1: Direct velocity control (recommended)
                navigate_to_wind_source(mc, logger)

                # Method 2: Turn and surge (alternative)
                #alternative_turn_and_surge(mc, logger)

            except KeyboardInterrupt:
                print("\nStopping...")
                mc.stop()
                mc.land()

            finally:
                print("Stopping logging...")
                logger.running = False
                logger.join()

                print("Sending DISARM request…")
                platform.send_arming_request(False)
                print("Disarmed. Program complete.")
