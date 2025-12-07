

import logging
from pynput import keyboard
import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.platformservice import PlatformService
import time
import logging
import time
import os
import csv
from datetime import datetime
from threading import Thread
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.platformservice import PlatformService
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.log import LogConfig
import numpy as np
URI = 'radio://0/80/2M'  # Change this to match your Crazyflie config

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)
URI = 'radio://0/80/2M/E7E7E7E7E7'   # <- change if needed
directory_path = r"C:\Users\ltjth\Documents\Research\UKF_Data"
base_filename = "CF_ARR_EKF1.0ms_MANUAL"
file_extension = ".csv"

# Speeds for square flights (m/s)
SPEEDS = [0.5]
SIDE_DISTANCE = 3.0  # meters
HEIGHT = 0.75        # takeoff height


# === FILE HANDLING ===
def get_log_filename():
    file_number = 1
    while True:
        csv_filename = f"{base_filename}{file_number}{file_extension}"
        full_path = os.path.join(directory_path, csv_filename)
        if not os.path.exists(full_path):
            return full_path
        file_number += 1


# === LOGGING THREAD ===
class LoggerThread(Thread):
    def __init__(self, cf, writer):
        super().__init__()
        self.cf = cf
        self.writer = writer
        self.running = True

        # Moving average filter for flow angle
        self.angle_window = []
        self.angle_window_size = 50  # Reduced for faster response

        # Calibration parameters
        self.calib_window = 400
        self.bx_window = []
        self.by_window = []
        self.bz_window = []
        self.gas_con_window = []
        self.mag_con_window = []
        self.mag_con_window_size = 20
        self.gas_con_window_size = 20  # Moving average window for gas concentration
        self.bx_offset = 0
        self.by_offset = 0
        self.gas_con_offset = 0
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
        self.gas_con = 0
        self.elapsed = 0

        # Drone Multiranger variables
        self.range_front = 0.0
        self.range_back = 0.0
        self.range_left = 0.0
        self.range_right = 0.0

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
    def moving_average_mag(self, new_mag):
        """Apply moving average filter to gas concentration"""
        self.mag_con_window.append(new_mag)
        if len(self.mag_con_window) > self.mag_con_window_size:
            self.mag_con_window.pop(0)
        return sum(self.mag_con_window) / len(self.mag_con_window)

    def moving_average_gas(self, new_gas):
        """Apply moving average filter to gas concentration"""
        self.gas_con_window.append(new_gas)
        if len(self.gas_con_window) > self.gas_con_window_size:
            self.gas_con_window.pop(0)
        return sum(self.gas_con_window) / len(self.gas_con_window)

    def run(self):
        # -------------------
        # LOG BLOCK 1: State + Wind
        # -------------------
        log_conf_1 = LogConfig(name='Logger', period_in_ms=10)  # 100 Hz
        log_conf_1.add_variable('stateEstimate.vx', 'float')
        log_conf_1.add_variable('stateEstimate.vy', 'float')
        log_conf_1.add_variable('stateEstimate.vz', 'float')
        log_conf_1.add_variable('stateEstimate.x', 'float')
        log_conf_1.add_variable('stateEstimate.y', 'float')
        log_conf_1.add_variable('windSensor.flowX', 'int16_t')
        log_conf_1.add_variable('windSensor.flowY', 'int16_t')
        log_conf_1.add_variable('windSensor.flowZ', 'int16_t')

        # -------------------
        # LOG BLOCK 2: Attitude + Ranges
        # -------------------
        log_conf_2 = LogConfig(name='Attitude', period_in_ms=10)
        log_conf_2.add_variable('stateEstimate.z', 'float')
        log_conf_2.add_variable('stateEstimate.qx', 'float')
        log_conf_2.add_variable('stateEstimate.qy', 'float')
        log_conf_2.add_variable('stateEstimate.qz', 'float')
        log_conf_2.add_variable('stateEstimateZ.ratePitch', 'int16_t')
        log_conf_2.add_variable('stateEstimateZ.rateRoll', 'int16_t')
        log_conf_2.add_variable('stateEstimateZ.rateYaw', 'int16_t')
        log_conf_2.add_variable('windSensor.gas', 'int16_t')
        #start = time.time()
        # -------------------
        # CALLBACKS
        # -------------------
        def log_data_1(timestamp, data, logconf):
            bx = -data['windSensor.flowX']
            by = -data['windSensor.flowY']
            bz = data['windSensor.flowZ']

            # Calibration
            if not self.calibrated:
                print("logger started")
                if len(self.bx_window) < self.calib_window:
                    self.bx_window.append(bx)
                    self.by_window.append(by)
                    self.bz_window.append(bz)
                    return
                else:
                    self.bx_offset = np.mean(self.bx_window)
                    self.by_offset = np.mean(self.by_window)
                    self.bz_offset = np.mean(self.bz_window)
                    self.calibrated = True
                    print(f"Calibration done: Bx offset={self.bx_offset:.2f}, By offset={self.by_offset:.2f}")

            # Timestamp
            now = datetime.now()
            #elapsed = start-elapsed
            # State values
            vx = data['stateEstimate.vx']
            vy = data['stateEstimate.vy']
            vz = data['stateEstimate.vz']
            px = data['stateEstimate.x']
            py = data['stateEstimate.y']


            # Calibrated whisker values
            bx_cal = bx - self.bx_offset
            by_cal = by - self.by_offset
            bz_cal = bz - self.bz_offset

            # Derived quantities
            flow_angle = np.degrees(np.arctan2(by_cal, bx_cal)) % 360
            flow_mag = np.sqrt(bx_cal ** 2 + by_cal ** 2)
            # flow_mag = self.moving_average_mag(flow_mag)
            # flow_angle = self.moving_average_angle(flow_angle)

            # Store most recent for merged CSV write
            self.latest_data_1 = {
                "Month": now.month,
                "Day": now.day,
                "Hour": now.hour,
                "Minute": now.minute,
                "Second": now.second,
                "Microsecond": now.microsecond,
                "Vx": vx, "Vy": vy, "Vz": vz,
                "Bx": bx_cal, "By": by_cal, "Bz": bz_cal,
                "FlowMag": flow_mag, "FlowAngle": flow_angle, "PosX": px, "PosY": py,
            }

        def log_data_2(timestamp, data, logconf):
            # Store attitude and range data
            self.latest_data_2 = {
                "PosZ": data['stateEstimate.z'],
                "Qx": data['stateEstimate.qx'],
                "Qy": data['stateEstimate.qy'],
                "Qz": data['stateEstimate.qz'],
                "PitchRate": data['stateEstimateZ.ratePitch'],
                "RollRate": data['stateEstimateZ.rateRoll'],
                "YawRate": data['stateEstimateZ.rateYaw'],"Gas": data['windSensor.gas']

            }

            # Merge and write to CSV if both data sets are available
            if hasattr(self, "latest_data_1") and self.running:
                row = {**self.latest_data_1, **self.latest_data_2}
                self.writer.writerow(row)

        def log_error(logconf, msg):
            print(f"Logging error in {logconf.name}: {msg}")

        # -------------------
        # REGISTER LOG CONFIGS
        # -------------------
        self.cf.log.add_config(log_conf_1)
        self.cf.log.add_config(log_conf_2)

        log_conf_1.data_received_cb.add_callback(log_data_1)
        log_conf_2.data_received_cb.add_callback(log_data_2)

        log_conf_1.error_cb.add_callback(log_error)
        log_conf_2.error_cb.add_callback(log_error)

        # -------------------
        # START LOGGING
        # -------------------
        try:
            log_conf_1.start()
            log_conf_2.start()

            while self.running:
                time.sleep(0.1)

            log_conf_1.stop()
            log_conf_2.stop()
        except Exception as e:
            print("Logging error:", e)

class KeyboardDrone:
    def __init__(self, mc):
        self.mc = mc
        self.velocity = 1.0
        self.ang_velocity = 120
        self.sleeptime = 0.5
        print('Press "u" for takeoff, "l" for land, SPACE for up, "c" for down.')

    def on_press(self, key):
        try:
            if key.char == 'w':
                self.mc.start_forward(self.velocity)
            if key.char == 'u':
                self.mc.take_off(0.7)
            if key.char == 's':
                self.mc.start_back(self.velocity)
            if key.char == 'a':
                self.mc.start_left(self.velocity)
            if key.char == 'd':
                self.mc.start_right(self.velocity)
            if key.char == 'c':
                self.mc.start_down(self.velocity)
            if key.char == 'l':
                self.mc.land()
            if key.char == 'q':
                self.mc.start_turn_left(self.ang_velocity)
            if key.char == 'e':
                self.mc.start_turn_right(self.ang_velocity)
        except AttributeError:
            # Handle special keys like space
            if key == keyboard.Key.space:
                self.mc.start_up(self.velocity)

    def on_release(self, key):
        self.mc.stop()


if __name__ == '__main__':
    cflib.crtp.init_drivers(enable_debug_driver=False)
    log_path = get_log_filename()

    with open(log_path, mode="w", newline='') as csv_file:
        fieldnames = [
            "Month", "Day", "Hour", "Minute", "Second", "Microsecond",
            "Vx", "Vy", "Vz",
            "Bx", "By", "Bz", "FlowMag", "FlowAngle", "PosX", "PosY", "PosZ",
            "Qx", "Qy", "Qz",
            "PitchRate", "RollRate", "YawRate","Gas"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        with SyncCrazyflie(URI) as scf:
            cf = scf.cf
            platform = PlatformService(cf)

            print("Arming...")
            platform.send_arming_request(True)

            try:
                with MotionCommander(scf) as mc:
                    drone = KeyboardDrone(mc)
                    time.sleep(2)
                    LoggerThread(cf, writer).start()
                    time.sleep(4)
                    with keyboard.Listener(on_press=drone.on_press,
                                           on_release=drone.on_release) as listener:
                        listener.join()
            except KeyboardInterrupt:
                print("Keyboard interrupt, landing...")
            finally:
                print("Disarming...")
                platform.send_arming_request(False)
                print("Disconnected.")
