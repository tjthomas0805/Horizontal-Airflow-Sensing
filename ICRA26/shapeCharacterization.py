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
URI = 'radio://0/80/2M/E7E7E7E7E7'   # <- change if needed
directory_path = r"C:\Users\ltjth\Documents\Research\VelocityLogs"
base_filename = "rightDiagonal0.5"
file_extension = ".csv"

# Speeds for square flights (m/s)
SPEEDS = [0.5]
SIDE_DISTANCE = 3.0  # meters
HEIGHT = 0.5        # takeoff height


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
    def __init__(self, cf, writer, speed):
        super().__init__()
        self.cf = cf
        self.writer = writer
        self.running = True
        self.current_speed = speed
        self.calib_window = 100
        self.bx_window = []
        self.by_window = []
        self.bx_offset = 0
        self.by_offset = 0
        self.calibrated = False

    def run(self):
        log_conf = LogConfig(name='Logger', period_in_ms=10)
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

            # --- Collect calibration window first ---
            if not self.calibrated:
                if len(self.bx_window) < self.calib_window:
                    self.bx_window.append(bx)
                    self.by_window.append(by)
                    return
                else:
                    self.bx_offset = sum(self.bx_window)/len(self.bx_window)
                    self.by_offset = sum(self.by_window)/len(self.by_window)
                    self.calibrated = True
                    print(f"Calibration done: Bx offset={self.bx_offset}, By offset={self.by_offset}")

            # --- Subtract offsets before writing ---
            bx_cal = bx - self.bx_offset
            by_cal = by - self.by_offset

            if self.running:
                now = datetime.now()
                self.writer.writerow({
                    "Month": now.month,
                    "Day": now.day,
                    "Hour": now.hour,
                    "Minute": now.minute,
                    "Second": now.second,
                    "Microsecond": now.microsecond,
                    "Speed": self.current_speed,
                    "Vx": data['stateEstimate.vx'],
                    "Vy": data['stateEstimate.vy'],
                    "PosX": data['stateEstimate.x'],
                    "PosY": data['stateEstimate.y'],
                    "PosZ": data['stateEstimate.z'],
                    "Bx": bx_cal,
                    "By": by_cal
                })

        def log_error(logconf, msg):
            print(f"Logging error: {msg}")

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


# def calibration(bx,by):
#     calibFlag = False
#     windowLength = 100
#     bx_window = []
#     by_window = []
#
#     while len(bx_window) < windowLength & len(by_window) < windowLength:
#         bx_window.append(bx)
#         by_window.append(by)
#
#     calibFlag = True
def diagonal(mc, speed):
    mc.start_linear_motion(speed, -speed, 0,0)
    time.sleep(6)
    mc.stop()
    time.sleep(2)
    mc.start_linear_motion(-speed, speed, 0, 0)
    time.sleep(6)
    mc.stop()

def spins(mc, speed):
    #mc.start_linear_motion(speed, 0, 0,60)
    mc.circle_left(0.5,speed,360.0)
    time.sleep(6)
    mc.stop()
def hover(mc: MotionCommander):
    mc.stop()
    time.sleep(10)
def forward_ramp_speed(mc: MotionCommander, target_speed: float, ramp_distance: float, update_rate=0.1):
    """
    Ramp forward speed from 0 to target_speed over ramp_distance.

    mc: MotionCommander object
    target_speed: final speed in m/s
    ramp_distance: distance in meters over which to ramp
    update_rate: time between speed updates in seconds
    """
    # Calculate number of steps
    total_steps = int(ramp_distance / (target_speed * update_rate / 2))  # approximate
    speeds = np.linspace(0, target_speed, total_steps)

    print(f"Ramping speed to {target_speed} m/s over {ramp_distance} m in {total_steps} steps")

    for s in speeds:
        #mc.start_forward(s)
        mc.start_right(s)
        time.sleep(update_rate)

    #mc.start_forward(target_speed)  # ensure final speed is exact
    mc.start_right(target_speed)

def cross(mc: MotionCommander, target_speed: float, ramp_distance: float, update_rate=0.1):
    """
    Ramp forward speed from 0 to target_speed over ramp_distance.

    mc: MotionCommander object
    target_speed: final speed in m/s
    ramp_distance: distance in meters over which to ramp
    update_rate: time between speed updates in seconds
    """
    # Calculate number of steps
    total_steps = int(ramp_distance / (target_speed * update_rate / 2))  # approximate
    speeds = np.linspace(0, target_speed, total_steps)


    print(f"Ramping speed to {target_speed} m/s over {ramp_distance} m in {total_steps} steps")

    for s in speeds:
        mc.start_forward(s)
        time.sleep(update_rate)
        if s >= target_speed:
            mc.stop()
            print("moving back in a sec")
            time.sleep(2)
            break
    for s in speeds:
        mc.start_back(s)
        time.sleep(update_rate)
        if s >= target_speed:
            mc.stop()
            print("leaving x dimension calibration")
            time.sleep(2)
            # mc.move_distance(1.5,0,0)
            break
    for s in speeds:
        mc.start_left(s)
        time.sleep(update_rate)
        if s >= target_speed:
            mc.stop()
            print("leaving x dimension calibration")
            time.sleep(2)
            break
    for s in speeds:
        mc.start_right(s)
        time.sleep(update_rate)
        if s >= target_speed:
            mc.stop()
            print("leaving x dimension calibration")
            time.sleep(2)
            break

def fly_circle(mc: MotionCommander, radius=3, velocity=0.8, steps=36):
    """
    Fly a circle by approximating it with small linear moves.
    Uses start_linear_motion() to move along each segment.
    mc: MotionCommander
    radius: circle radius (m)
    velocity: tangential velocity (m/s)
    steps: number of linear segments in the circle
    """
    import math
    arc_length = 2 * math.pi * radius
    duration = arc_length / velocity
    dt = duration / steps
    segment_length = arc_length / steps

    print(f"\n--- Flying circle with linear segments: radius {radius} m,"
          f" velocity {velocity} m/s, {steps} segments ---")

    for i in range(steps):
        angle = 2 * math.pi * i / steps
        dx = -segment_length * math.sin(angle)  # X component
        dy =  segment_length * math.cos(angle)  # Y component
        mc.start_linear_motion(dx, dy, 0)
        time.sleep(dt)

    mc.stop()
    time.sleep(1)



    #mc.start_forward(target_speed)  # ensure final speed is exact
    #mc.start_right(target_speed)
# === FLIGHT PATTERN ===
def fly_square(mc, speed, distance):
    """Fly a square trajectory of given distance at given speed."""
    duration = distance / speed
    print(f"\n--- Flying square: {distance} m per side at {speed} m/s "
          f"(time {duration:.1f} s per side) ---")

    mc.start_forward(speed)
    time.sleep(duration+2)
    mc.stop()
    time.sleep(2)

    mc.start_right(speed)
    time.sleep(duration+2)
    mc.stop()
    time.sleep(2)

    mc.start_back(speed)
    time.sleep(duration+2)
    mc.stop()
    time.sleep(2)

    mc.start_left(speed)
    time.sleep(duration+2)
    mc.stop()
    time.sleep(2)

# def fly_diagonal(mc, speed, distance):
#     mc.


# === MAIN PROGRAM ===
if __name__ == '__main__':
    cflib.crtp.init_drivers(enable_debug_driver=False)
    log_path = get_log_filename()

    with open(log_path, mode="w", newline='') as csv_file:
        fieldnames = [
            "Month","Day","Hour","Minute","Second","Microsecond",
            "Speed","Vx","Vy","PosX","PosY","PosZ","Bx","By"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            cf = scf.cf
            platform = PlatformService(crazyflie=cf)

            print("Connected — sending ARM request…")
            platform.send_arming_request(True)
            print("Armed!")

            mc = MotionCommander(scf, default_height=HEIGHT)

            # === KILLSWITCH HANDLER ===
            def killswitch():
                print("\nESC pressed! Landing immediately...")
                mc.land()
                time.sleep(2)
                platform.send_arming_request(False)
                print("Disarmed. Exiting program.")
                os._exit(0)  # Force exit immediately

            keyboard.add_hotkey('esc', killswitch)

            # Take off
            print("Taking off...")
            mc.take_off(HEIGHT)
            time.sleep(2)

            # Perform square flights at each speed
            for speed in SPEEDS:
                logger = LoggerThread(cf, writer, speed)
                logger.start()
                #cross(mc,target_speed=1.0, ramp_distance=3.0 )
                #fly_circle(mc)
                #forward_ramp_speed(mc, target_speed=1.0, ramp_distance=3.0)
                #fly_square(mc,1.0,3)
                #hover(mc)
                #spins(mc,0.5)
                diagonal(mc,0.5)
                time.sleep(2)
                print("Completed calibration, landing...")
                # mc.stop()
                mc.land()
                time.sleep(3)
                break# pause between squares



            # Land normally if killswitch was not pressed
            print("Landing...")
            mc.land()
            time.sleep(2)

            print("Sending DISARM request…")
            platform.send_arming_request(False)
            print("Disarmed.")