import logging
import time
import os
import csv
from datetime import datetime
from threading import Thread
import keyboard
import cflib.crtp
# from CFLogger import *
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.platformservice import PlatformService
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.log import LogConfig
import numpy as np

# === CONFIGURATION ===
URI = 'radio://0/80/2M/E7E7E7E7E7'  # <-- change if needed
directory_path = r"C:\Users\bridg\OneDrive\Documents\Research\ICRA26\Turn_Test_Data"
base_filename = "castTest"
file_extension = ".csv"

# Speeds for square flights (m/s)
SPEEDS = [0.5]
SIDE_DISTANCE = 3.0  # meters
HEIGHT = 1  # takeoff height

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
min_flow_threshold = 15
angle_threshold = 10
desired_flow_angle = 180
flowMap = []
gas_gradient = []
# once you detect flow add these as a tuple to flowMap

# Wind source navigation parameters
SEARCH_SPEED = 0.3
SEARCH_SPEED_X = 0.2
SEARCH_SPEED_Y = 0.4
SEARCH_DURATION = 0.7  # seconds per search segment
APPROACH_SPEED = 0.5
APPROACH_SPEED_X = 0.5
APPROACH_SPEED_Y = 1
MAX_FLOW_THRESHOLD = 90  # When to consider we've reached the source
LOCAL_MAX_FLOW = 0  # Parameter used for Cast and Surge Algorithm
TURN_RATE = 45  # degrees/second for turning towards source

# Cast and Surge Algorithm Parameters
FORWARD_MOVE_DURATION = 1  # seconds
CAST_DURATION = 2  # seconds
CAST_DISTANCE = 1.5  # meters for unilateral cast
CAST_DISTANCE_X = 0.3
CAST_DISTANCE_Y = 1

# Test Parameters
TEST_DISTANCE = 1.5  # meters for state estimator test

# Multiranger Parameters
MIN_RANGER_DISTANCE = 1000  # mm, minimum distance to obstacle

# Gas Sensor Parameters
GAS_THRESHOLD = 8  # Threshold for gas concentration
local_gas_concentration = []
gas_gradients = []

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
        self.angle_window_size = 5  # Reduced for faster response

        # Calibration parameters
        self.calib_window = 100
        self.bx_window = []
        self.by_window = []
        self.gas_con_window = []
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

    def moving_average_gas(self, new_gas):
        """Apply moving average filter to gas concentration"""
        self.gas_con_window.append(new_gas)
        if len(self.gas_con_window) > self.gas_con_window_size:
            self.gas_con_window.pop(0)
        return sum(self.gas_con_window) / len(self.gas_con_window)
    
    def run(self):
        log_conf = LogConfig(name='Logger', period_in_ms=10)  # 20 Hz
        log_conf.add_variable('stateEstimate.vx', 'float')
        log_conf.add_variable('stateEstimate.vy', 'float')
        log_conf.add_variable('stateEstimate.x', 'float')
        log_conf.add_variable('stateEstimate.y', 'float')
        log_conf.add_variable('windSensor.flowX', 'int16_t')
        log_conf.add_variable('windSensor.flowY', 'int16_t')
        log_conf.add_variable('windSensor.gas', 'int16_t')
        
        log_conf_multi = LogConfig(name='Multiranger_Logger', period_in_ms=10)  # 10 Hz
        log_conf_multi.add_variable('range.front', 'float')
        log_conf_multi.add_variable('range.back', 'float')
        log_conf_multi.add_variable('range.left', 'float')
        log_conf_multi.add_variable('range.right', 'float')

        def log_data(timestamp, data, logconf):
            bx = data['windSensor.flowX']
            by = data['windSensor.flowY']
            gas_con = data['windSensor.gas']

            # Calibration step
            if not self.calibrated:
                if len(self.bx_window) < self.calib_window:
                    self.bx_window.append(bx)
                    self.by_window.append(by)
                    self.gas_con_window.append(gas_con)
                    return
                else:
                    self.bx_offset = sum(self.bx_window) / len(self.bx_window)
                    self.by_offset = sum(self.by_window) / len(self.by_window)
                    self.gas_con_offset = sum(self.gas_con_window) / len(self.gas_con_window)
                    self.calibrated = True
                    print(
                        f"Calibration done: Bx offset={self.bx_offset:.2f}, By offset={self.by_offset:.2f}, Gas offset={self.gas_con_offset:.2f}")

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
            gas_cal = gas_con - self.gas_con_offset

            # Update flow measurements
            self.latest_bx = bx_cal
            self.latest_by = by_cal
            self.gas_con = gas_cal

            # Calculate flow angle and magnitude
            raw_angle = ((np.arctan2(by_cal, bx_cal) * (180 / np.pi)) - 180) % 360
            gas_con = self.gas_con
            self.gas_con = self.moving_average_gas(gas_con)
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
                    "Bx": bx_cal,
                    "By": by_cal,
                    "Flow Mag": self.flowMag,
                    "Flow Angle": self.flowAngle,
                    "Gas Concentration": data['windSensor.gas'],
                    "Range Front": self.range_front,
                    "Range Back": self.range_back,
                    "Range Left": self.range_left,
                    "Range Right": self.range_right,
                })

        def log_multi_data(timestamp, data, logconf):
            self.range_front = data['range.front']
            self.range_back = data['range.back']
            self.range_left = data['range.left']
            self.range_right = data['range.right']
            

        def log_error(logconf, msg):
            print(f"Logging error: {msg}")

        # Start logging
        self.cf.log.add_config(log_conf)
        self.cf.log.add_config(log_conf_multi)

        log_conf.data_received_cb.add_callback(log_data)
        log_conf_multi.data_received_cb.add_callback(log_multi_data)

        log_conf.error_cb.add_callback(log_error)
        log_conf_multi.error_cb.add_callback(log_error)
        try:
            log_conf.start()
            log_conf_multi.start()

            while self.running:
                time.sleep(0.1)

            log_conf.stop()
            log_conf_multi.stop()
        except Exception as e:
            print("Logging error:", e)



# === NAVIGTION MODES ===

def navigate_to_wind_source(mc, logger):
    """
    Main navigation function to find and approach wind source
    """
    print("Starting wind source navigation...")

    # Phase 1: Search for wind
    search_for_wind(mc, logger)

    # Phase 2: Navigate towards source
    approach_wind_source(mc, logger)


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


def cast_and_surge(mc, logger):
    # Do Zig-Zag Motions to search for wind
    seek_and_turn(mc, logger)

    # Go Towards Flow once Detected
    unilateralSurge(mc, logger)


def spiralSearch(mc, logger):
    # Do Spiral Motions to  search for wind
    seek_and_turn(mc, logger)

    # Go Towards Flow once Detected
    spiralSurge(mc, logger)


def gradientSearch(mc, logger):
    # Cast out to search for gas
    search_for_gas(mc, logger)

    # Detect local gradients to build trajectory
    gradientSurge(mc, logger)



# === CAPABILITY TEST MODES ===

def spiralTest(mc, logger):
    # Testing Spiral Motion
    print("Starting spiral motion test...")
    spiralMove(mc, logger)


def inFlightTurnTest(mc, logger):
    global previous_error, integral, previous_time, current_time, drone_heading, flowMap, desired_flow_angle, min_flow_threshold, angle_threshold, Kp, Ki, Kd, last_error
    while True:
        checkRangers(logger)
        if logger.calibrated:  # Ensure the sensor is calibrated
            print(f"Bx: {logger.latest_bx:.2f}, By: {logger.latest_by:.2f}, Drone X: {logger.droneX}, Drone Y: {logger.droneY}, Flow Magnitude: {logger.flowMag}")
            while logger.flowMag >= min_flow_threshold:
                checkRangers(logger)
                #print(f"Flow angle at: {logger.flowAngle}, Flow Magnitude is: {logger.flowMag}")
                flowMap.append((logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))
                error = desired_flow_angle - logger.flowAngle
                current_time = logger.microsecond
                if previous_time is None or previous_time == 0: previous_time = current_time - 0.1
                delta_time = current_time - previous_time
                integral += error * delta_time
                turn_command = int(Kp * error + Ki * integral)
                previous_error = error
                previous_time = current_time

                if abs(error) <= angle_threshold:  # if error is within desired threshold, hover
                    print(f"Within Threshold. Moving towards detected flow at {logger.flowAngle} degrees with a magnitude of {logger.flowMag}")
                    mc.stop() # Start Non-Blocking Turn
                    time.sleep(0.05)
                    flowMap.append((logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))

                elif (270 <= logger.flowAngle <= 360) or ((desired_flow_angle + angle_threshold) <= logger.flowAngle <= 270):
                    right_command = abs(turn_command)/100  # Convert to angular rate in degrees/s, scale down
                    right_command = max(min(right_command, 1.0), 0)  # Scale to Crazyflie range
                    print(f"Turning left, source angle & Magnitude: {logger.flowAngle} | {logger.flowMag} ; error: {error} ; command:{right_command}")
                    mc.start_turn_left(right_command * 90)  # Convert to appropriate angular rate
                    time.sleep(0.05)
                    flowMap.append((logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))

                elif (90 <= logger.flowAngle <= (desired_flow_angle - angle_threshold)) or (0 <= logger.flowAngle <= 90):
                    left_command = abs(turn_command)/100  # Convert to angular rate in degrees/s, scale down
                    left_command = max(min(left_command, 1.0), 0)  # Scale to Crazyflie range
                    left_command = abs(left_command)  # Make positive for turn_left function
                    print(f"Turning right, source angle & Magnitude: {logger.flowAngle} | {logger.flowMag} ; error:{error} ; command:{left_command}")
                    mc.start_turn_right(left_command * 90)  # Convert to appropriate angular rate
                    time.sleep(0.05)
                    flowMap.append((logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))
                    
                last_error = error
        else:
            print("Calibrating...")
            time.sleep(0.2)


def gradientGasDetectionTest(mc, logger):
    global flowMap
    print("Starting gradient gas detection test...")
    search_for_wind(mc, logger)


def sensorResponseTest(mc, logger):
    global previous_error, integral, previous_time, current_time, drone_heading, flowMap, desired_flow_angle, min_flow_threshold, angle_threshold, Kp, Ki, Kd, last_error
    while True:
        if logger.calibrated:  # Ensure the sensor is calibrated
            print(f"Bx: {logger.latest_bx:.2f}, By: {logger.latest_by:.2f}, Drone X: {logger.droneX}, Drone Y: {logger.droneY}, Flow Magnitude: {logger.flowMag}")
            checkRangers(logger)
            while logger.flowMag >= min_flow_threshold:
                #print(f"Flow angle at: {logger.flowAngle}, Flow Magnitude is: {logger.flowMag}")
                flowMap.append((logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))
                error = desired_flow_angle - logger.flowAngle
                current_time = logger.microsecond
                if previous_time is None or previous_time == 0: previous_time = current_time - 0.1
                delta_time = current_time - previous_time
                integral += error * delta_time
                turn_command = int(Kp * error + Ki * integral)
                previous_error = error
                previous_time = current_time

                if abs(error) <= angle_threshold:  # if error is within desired threshold, hover
                    print(f"Within Threshold. Moving towards detected flow at {logger.flowAngle} degrees with a magnitude of {logger.flowMag}")
                    time.sleep(0.05)
                    flowMap.append((logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))

                elif (270 <= logger.flowAngle <= 360) or ((desired_flow_angle + angle_threshold) <= logger.flowAngle <= 270):
                    right_command = abs(turn_command)/100  # Convert to angular rate in degrees/s, scale down
                    right_command = max(min(right_command, 1.0), 0)  # Scale to Crazyflie range
                    print(f"Turning left, source angle & Magnitude: {logger.flowAngle} | {logger.flowMag} ; error: {error} ; command:{right_command}")
                    time.sleep(0.05)
                    flowMap.append((logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))

                elif (90 <= logger.flowAngle <= (desired_flow_angle - angle_threshold)) or (0 <= logger.flowAngle <= 90):
                    left_command = abs(turn_command)/100  # Convert to angular rate in degrees/s, scale down
                    left_command = max(min(left_command, 1.0), 0)  # Scale to Crazyflie range
                    left_command = abs(left_command)  # Make positive for turn_left function
                    print(f"Turning right, source angle & Magnitude: {logger.flowAngle} | {logger.flowMag} ; error:{error} ; command:{left_command}")
                    time.sleep(0.05)
                    flowMap.append((logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))
                    
                last_error = error
        else:
            print("Calibrating...")
            time.sleep(0.2)


def stateEstimatorTest(mc, logger):
    # Cast Left
    print("Casting Left...")
    start_posX = logger.droneX
    start_posY = logger.droneY
    mc.start_linear_motion(SEARCH_SPEED_X, SEARCH_SPEED_Y, 0)

    while logger.droneX < CAST_DISTANCE_X + start_posX and logger.droneY < CAST_DISTANCE_Y + start_posY:
        print(logger.droneX)
        continue
    mc.stop()
    time.sleep(0.5)
    # Cast Right
    print("Casting Right...")
    start_posX = logger.droneX
    start_posY = logger.droneY
    mc.start_linear_motion(SEARCH_SPEED_X, -SEARCH_SPEED_Y, 0)
    while logger.droneX < CAST_DISTANCE_X + start_posX and logger.droneY > -CAST_DISTANCE_Y + start_posY:
        print(logger.droneX)
        continue
    mc.stop()
    time.sleep(0.5)


def multirangerTest(mc, logger):
    # Testing Multiranger (Obstacle Avoidance)
    print("Starting multiranger test...")
    while True:
        checkRangers(logger)



# === INITIAL FLOW DISCOVERY FUNCTIONS ===

# Search for gas feducial
def search_for_gas(mc, logger):
    global flowMap, CAST_DISTANCE_X, CAST_DISTANCE_Y, local_gas_concentration, gas_gradients
    """
    Search phase - move forward slowly until gas is detected
    """

    for _ in range(5):  # Limit to 5 iterations for testing``
        # Cast Left - check thresholds during movement
        start_posX = logger.droneX
        start_posY = logger.droneY
        checkRangers(logger)
        mc.start_linear_motion(SEARCH_SPEED_X, SEARCH_SPEED_Y, 0)

        while (logger.droneX < CAST_DISTANCE_X + start_posX and
            logger.droneY < CAST_DISTANCE_Y + start_posY):
            # Check thresholds during casting - exit only if BOTH exceed threshold
            flowMap.append((logger.droneX, logger.droneY, logger.gas_con, logger.flowAngle))
            local_gas_concentration.append((logger.droneX, logger.droneY, logger.gas_con))
            print(f"Casting Left - Wind: {logger.flowMag:.2f}, Gas: {logger.gas_con}")
            time.sleep(0.01)

        mc.stop()
        time.sleep(0.5)
        return_x = logger.droneX - (CAST_DISTANCE_X/2) 
        return_y = logger.droneX - (CAST_DISTANCE_Y/2)
        return_gas_con = max(local_gas_concentration[:][2])
        dx, dy = calculate_spatial_gradient(local_gas_concentration)

        # Calculate Local Gas Concentration Gradient for First Cast
        if dx is not None:
            gradient_magnitude = np.sqrt(dx**2 + dy**2)
            gradient_angle = (np.degrees(np.arctan2(dy, dx)))%360
            gas_gradients.append((return_x, return_y, return_gas_con, gradient_magnitude))
            print(f"Local gradient: magnitude={gradient_magnitude:.3f}, angle={gradient_angle:.1f}°")

        local_gas_concentration = []  # Reset for next cast

        # Cast Right - check thresholds during movement
        start_posX = logger.droneX
        start_posY = logger.droneY
        checkRangers(logger)
        mc.start_linear_motion(SEARCH_SPEED_X, -SEARCH_SPEED_Y, 0)

        while (logger.droneX < CAST_DISTANCE_X + start_posX and
               logger.droneY > -CAST_DISTANCE_Y + start_posY):
            # Check thresholds during casting - exit only if BOTH exceed threshold
            flowMap.append((logger.droneX, logger.droneY, logger.gas_con, logger.flowAngle))
            print(f"Casting Right - Wind: {logger.flowMag:.2f}, Gas: {logger.gas_con}")
            time.sleep(0.01)

        mc.stop()
        time.sleep(0.5)

        return_x = logger.droneX - (CAST_DISTANCE_X/2) 
        return_y = logger.droneX - (CAST_DISTANCE_Y/2)
        return_gas_con = max(local_gas_concentration[:][2])
        dx, dy = calculate_spatial_gradient(local_gas_concentration)

        # Calculate Local Gas Concentration Gradient for First Cast
        if dx is not None:
            gradient_magnitude = np.sqrt(dx**2 + dy**2)
            gradient_angle = (np.degrees(np.arctan2(dy, dx)))%360
            gas_gradients.append((return_x, return_y, return_gas_con, gradient_magnitude))
            print(f"Local gradient: magnitude={gradient_magnitude:.3f}, angle={gradient_angle:.1f}°")
        
        # Reset for the next cast
        local_gas_concentration = []  # Reset for next cast

        # Expand search area for next iteration
        CAST_DISTANCE_X += 0.3
        CAST_DISTANCE_Y += 0.3


# Generic Zig-Zag Cast/Searching Motion
def search_for_wind(mc, logger):
    global flowMap, CAST_DISTANCE_X, CAST_DISTANCE_Y
    """
    Search phase - move forward slowly until wind is detected
    """
    time.sleep(1)

    # Keep AND logic - continue while BOTH are below threshold
    while (logger.flowMag <= min_flow_threshold): #and (abs(logger.gas_con) <= GAS_THRESHOLD):
        print(f"Searching - Wind: {logger.flowMag:.2f}, Gas: {logger.gas_con}")
        flowMap.append(
            (logger.droneX, logger.droneY, logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))

        # Cast Left - check thresholds during movement
        start_posX = logger.droneX
        start_posY = logger.droneY
        checkRangers(logger)
        mc.start_linear_motion(SEARCH_SPEED_X, SEARCH_SPEED_Y, 0)

        while (logger.droneX < CAST_DISTANCE_X + start_posX and
               logger.droneY < CAST_DISTANCE_Y + start_posY):
            # Check thresholds during casting - exit only if BOTH exceed threshold
            if (logger.flowMag > min_flow_threshold):# and (abs(logger.gas_con) > GAS_THRESHOLD):
                print(f"Both thresholds crossed during left cast! Wind: {logger.flowMag:.2f}, Gas: {logger.gas_con}")
                mc.stop()
                return  # Exit the function immediately

            print(f"Casting Left - Wind: {logger.flowMag:.2f}, Gas: {logger.gas_con}")
            time.sleep(0.01)

        mc.stop()
        time.sleep(0.5)

        # Check again before right cast - exit only if BOTH exceed threshold
        if (logger.flowMag > min_flow_threshold):# and (abs(logger.gas_con) > GAS_THRESHOLD):
            print(f"Both thresholds crossed between casts! Wind: {logger.flowMag:.2f}, Gas: {logger.gas_con}")
            return

        # Cast Right - check thresholds during movement
        start_posX = logger.droneX
        start_posY = logger.droneY
        checkRangers(logger)
        mc.start_linear_motion(SEARCH_SPEED_X, -SEARCH_SPEED_Y, 0)

        while (logger.droneX < CAST_DISTANCE_X + start_posX and
               logger.droneY > -CAST_DISTANCE_Y + start_posY):
            # Check thresholds during casting - exit only if BOTH exceed threshold
            if (logger.flowMag > min_flow_threshold):# and (abs(logger.gas_con) > GAS_THRESHOLD):
                print(f"Both thresholds crossed during right cast! Wind: {logger.flowMag:.2f}, Gas: {logger.gas_con}")
                mc.stop()
                return  # Exit the function immediately

            print(f"Casting Right - Wind: {logger.flowMag:.2f}, Gas: {logger.gas_con}")
            time.sleep(0.01)

        mc.stop()
        time.sleep(0.5)

        # Expand search area for next iteration
        CAST_DISTANCE_X += 0.3
        CAST_DISTANCE_Y += 0.3

    #print(f"Both thresholds detected! Wind: {logger.flowMag:.2f} | Gas: {logger.gas_con}")
    # mc.stop()
    # time.sleep(0.01)

# Zig-Zag Cast/Searching Motion with Immediate Turn to Source
def seek_and_turn(mc, logger):
    global flowMap, CAST_DISTANCE_X, CAST_DISTANCE_Y
    """
    Search phase - move forward slowly until wind is detected
    """
    time.sleep(1)

    # Keep AND logic - continue while BOTH are below threshold
    while (logger.flowMag <= min_flow_threshold) and (abs(logger.gas_con) <= GAS_THRESHOLD):
        print(f"Searching - Wind: {logger.flowMag:.2f}, Gas: {logger.gas_con}")
        flowMap.append(
            (logger.droneX, logger.droneY, logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))

        # Cast Left - check thresholds during movement
        start_posX = logger.droneX
        start_posY = logger.droneY
        checkRangers(logger)
        mc.start_linear_motion(SEARCH_SPEED_X, SEARCH_SPEED_Y, 0)

        while (logger.droneX < CAST_DISTANCE_X + start_posX and
               logger.droneY < CAST_DISTANCE_Y + start_posY):
            # Check thresholds during casting - exit only if BOTH exceed threshold
            if (logger.flowMag > min_flow_threshold) and (abs(logger.gas_con) > GAS_THRESHOLD):
                print(f"Both thresholds crossed during left cast! Wind: {logger.flowMag:.2f}, Gas: {logger.gas_con}")
                mc.stop()
                turnToSource(mc, logger)  # Immediately turn to source
                return  # Exit the function immediately

            print(f"Casting Left - Wind: {logger.flowMag:.2f}, Gas: {logger.gas_con}")
            time.sleep(0.01)

        mc.stop()
        time.sleep(0.5)

        # Check again before right cast - exit only if BOTH exceed threshold
        if (logger.flowMag > min_flow_threshold) and (abs(logger.gas_con) > GAS_THRESHOLD):
            print(f"Both thresholds crossed between casts! Wind: {logger.flowMag:.2f}, Gas: {logger.gas_con}")
            return

        # Cast Right - check thresholds during movement
        start_posX = logger.droneX
        start_posY = logger.droneY
        checkRangers(logger)
        mc.start_linear_motion(SEARCH_SPEED_X, -SEARCH_SPEED_Y, 0)

        while (logger.droneX < CAST_DISTANCE_X + start_posX and
               logger.droneY > -CAST_DISTANCE_Y + start_posY):
            # Check thresholds during casting - exit only if BOTH exceed threshold
            if (logger.flowMag > min_flow_threshold) and (abs(logger.gas_con) > GAS_THRESHOLD):
                print(f"Both thresholds crossed during right cast! Wind: {logger.flowMag:.2f}, Gas: {logger.gas_con}")
                mc.stop()
                turnToSource(mc, logger)  # Immediately turn to source
                return  # Exit the function immediately

            print(f"Casting Right - Wind: {logger.flowMag:.2f}, Gas: {logger.gas_con}")
            time.sleep(0.01)

        mc.stop()
        time.sleep(0.5)

        # Expand search area for next iteration
        CAST_DISTANCE_X += 0.3
        CAST_DISTANCE_Y += 0.3



# === CAST/BACKTRACK FUNCTIONS ===

# Spiral Search: Expanding Spiral Motion Until Wind is Detected
def spiralCast(mc, logger):
    """
    Search phase - move in an expanding spiral until wind is detected
    """
    print("Searching for wind with spiral cast...")
    radius = 0.5  # initial radius
    angle = 0
    angle_increment = 10  # degrees
    radius_increment = 0.1  # meters

    while logger.flowMag < LOCAL_MAX_FLOW:
        checkRangers(logger)
        # Calculate target position in spiral
        rad_angle = np.radians(angle)
        target_x = radius * np.cos(rad_angle)
        target_y = radius * np.sin(rad_angle)

        # Move to target position
        mc.start_linear_motion(target_x, target_y, 0)
        time.sleep(0.3)  # Adjust time to control speed

        # Update angle and radius for next point in spiral
        angle += angle_increment
        if angle >= 360:
            angle = angle % 360
            radius += radius_increment
        print(f"No wind detected (mag: {logger.flowMag:.2f}), continuing spiral search...")

    LOCAL_MAX_FLOW = logger.flowMag
    print(f"Wind detected! Magnitude: {logger.flowMag:.2f}")
    mc.stop()
    time.sleep(0.2)


# Cast and Surge: Unilateral Casting Motion For Surging Towards Source
def unilateralCast(mc, logger):
    """
    Search phase - move forward slowly until wind is detected
    """
    print("Searching for increased flow signal...")
    time.sleep(1)
    while logger.flowMag < LOCAL_MAX_FLOW:
        print(f"No wind detected (mag: {logger.flowMag:.2f}), continuing search...")
        flowMap.append(
            (logger.droneX, logger.droneY, logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))

        # Cast Left
        startPosX = logger.droneX
        startPosY = logger.droneY
        checkRangers(logger)
        mc.start_linear_motion(0, SEARCH_SPEED_Y, 0)
        while logger.droneY < CAST_DISTANCE + startPosY:
            if (logger.flowMag > min_flow_threshold) and (abs(logger.gas_con) > GAS_THRESHOLD):
                print(f"Both thresholds crossed during right cast! Wind: {logger.flowMag:.2f}, Gas: {logger.gas_con}")
                mc.stop()
                turnToSource(mc, logger)  # Immediately turn to source
                return  # Exit the function immediately 
        mc.stop()

        # Cast Right
        startPosX = logger.droneX
        startPosY = logger.droneY
        checkRangers(logger)
        mc.start_linear_motion(0, -SEARCH_SPEED_Y, 0)
        while logger.droneY > -CAST_DISTANCE + startPosY:
            if (logger.flowMag > min_flow_threshold) and (abs(logger.gas_con) > GAS_THRESHOLD):
                print(f"Both thresholds crossed during right cast! Wind: {logger.flowMag:.2f}, Gas: {logger.gas_con}")
                mc.stop()
                turnToSource(mc, logger)  # Immediately turn to source
                return  # Exit the function immediately
            time.sleep(0.01)
        mc.stop()

        FORWARD_MOVE_DURATION += 1
        CAST_DURATION += 1

    LOCAL_MAX_FLOW = logger.flowMag
    print(f"Wind detected! Magnitude: {logger.flowMag:.2f}")
    mc.stop()
    time.sleep(0.2)



# === SURGE FUNCTIONS ===

# Vector Surge: Direct Velocity Control Towards Source
def approach_wind_source(mc, logger):
    global flowMap
    """
    Approach phase - navigate towards wind source using flow direction
    """
    print("Approaching wind source...")

    while logger.flowMag < MAX_FLOW_THRESHOLD:
        if logger.flowMag <= min_flow_threshold:
            print("Lost wind signal, resuming search...")
            search_for_wind(mc, logger)
            continue

        # Calculate wind source direction (opposite to flow direction)
        source_angle = (logger.flowAngle - 180) % 360
        # np.gradient(flowMap[:][0:3])
        flowMap.append(
            (logger.droneX, logger.droneY, logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))
        # Convert to velocity components (NED frame)
        # 0° = North, 90° = East, 180° = South, 270° = West
        vx = APPROACH_SPEED * np.cos(np.radians(source_angle))  # North component
        vy = APPROACH_SPEED * np.sin(np.radians(source_angle))  # East component

        print(f"Gas: {logger.gas_con:.1f} Flow: {logger.flowMag:.2f} @ {logger.flowAngle:.1f}° | "
              f"Source: {source_angle:.1f}° | "
              f"Velocity: vx={vx:.2f}, vy={vy:.2f}")

        # Move towards source
        checkRangers(logger)
        mc.start_linear_motion(vx, vy, 0)

    print(f"Reached wind source! Flow magnitude: {logger.flowMag:.2f}")
    mc.stop()

    # Land at source
    print("Landing at wind source...")
    mc.start_down(0.3)
    time.sleep(2)
    mc.land()


# Cast and Surge: Surge with Frequent Flow Checks for Flow Navigation
def unilateralSurge(mc, logger):
    global flowMap
    """
    Approach phase - navigate towards wind source using flow direction
    """
    print("Approaching wind source...")

    while logger.flowMag < MAX_FLOW_THRESHOLD:
        if logger.flowMag < min_flow_threshold:
            print("Lost wind signal, resuming search...")
            unilateralCast(mc, logger)
            continue

        # Calculate wind source direction (opposite to flow direction)
        source_angle = (logger.flowAngle + 180) % 360
        # np.gradient(flowMap[:][0:3])
        flowMap.append(
            (logger.droneX, logger.droneY, logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))
        # Convert to velocity components (NED frame)
        # 0° = North, 90° = East, 180° = South, 270° = West
        vx = APPROACH_SPEED * np.cos(np.radians(source_angle))  # North component
        vy = APPROACH_SPEED * np.sin(np.radians(source_angle))  # East component

        print(f"Flow: {logger.flowMag:.2f} @ {logger.flowAngle:.1f}° | "
              f"Source: {source_angle:.1f}° | "
              f"Velocity: vx={vx:.2f}, vy={vy:.2f}")

        # Move towards source
        mc.start_forward(APPROACH_SPEED)
        time.sleep(0.1)

    print(f"Reached wind source! Flow magnitude: {logger.flowMag:.2f}")
    mc.stop()
    time.sleep(1)

    # Land at source
    print("Landing at wind source...")
    mc.start_down(0.3)
    time.sleep(2)
    mc.land()


# Spiral Surge:
def spiralSurge(mc, logger):
    global flowMap
    """
    Approach phase - navigate towards wind source using flow direction
    """
    print("Approaching wind source...")

    while logger.flowMag < MAX_FLOW_THRESHOLD:
        if logger.flowMag > LOCAL_MAX_FLOW:
            LOCAL_MAX_FLOW = logger.flowMag

        if logger.flowMag < min_flow_threshold:
            print("Lost wind signal, resuming search...")
            spiralCast(mc, logger)
            continue

        # Calculate wind source direction (opposite to flow direction)
        source_angle = (logger.flowAngle + 180) % 360
        # np.gradient(flowMap[:][0:3])
        flowMap.append(
            (logger.droneX, logger.droneY, logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))
        # Convert to velocity components (NED frame)
        # 0° = North, 90° = East, 180° = South, 270° = West
        vx = APPROACH_SPEED * np.cos(np.radians(source_angle))  # North component
        vy = APPROACH_SPEED * np.sin(np.radians(source_angle))  # East component

        print(f"Flow: {logger.flowMag:.2f} @ {logger.flowAngle:.1f}° | "
              f"Source: {source_angle:.1f}° | "
              f"Velocity: vx={vx:.2f}, vy={vy:.2f}")

        # Move towards source
        checkRangers(logger)
        mc.start_forward(APPROACH_SPEED)
        time.sleep(0.1)

    print(f"Reached wind source! Flow magnitude: {logger.flowMag:.2f}")
    mc.stop()
    time.sleep(1)

    # Land at source
    print("Landing at wind source...")
    mc.start_down(0.3)
    time.sleep(2)
    mc.land()


# Gradient Surge:
def gradientSurge(mc, logger):
    global flowMap, gas_gradients
    
    gas_data = gas_gradients.copy()  # Copy existing data points
    gas_x = [point[0] for point in gas_data]
    gas_y = [point[1] for point in gas_data]
    gas_concentration = [point[2] for point in gas_data]
    gas_gradient_magnitude = [point[3] for point in gas_data]
    
    gradient_threshold = 0.001  # Minimum gradient magnitude to continue
    
    mc.move_to(gas_x, gas_y, 1.0, 0.5)  # Move to last known gas gradient position
    time.sleep(2)
    
    print(f"Final position: ({logger.droneX:.2f}, {logger.droneY:.2f})")
    print(f"Final gas concentration: {logger.gas_con}")
    
    # Land at estimated source location
    mc.stop()
    time.sleep(1)
    print("Landing at estimated source location...")



# === HELPER FUNCTIONS ===

# Calculate Gradient of Local Gas Concentration
def calculate_spatial_gradient(positions_concentrations, window_size=200):
    """
    Calculate 2D gradient from spatial concentration data
    Returns gradient vectors (dx, dy) pointing toward steepest ascent
    """
    if len(positions_concentrations) < window_size:
        return None, None
    
    # Extract recent data
    recent_data = positions_concentrations[-window_size:]
    
    # Separate into arrays
    x_coords = np.array([point[0] for point in recent_data])
    y_coords = np.array([point[1] for point in recent_data])
    concentrations = np.array([point[2] for point in recent_data])
    
    # Calculate gradients using finite differences
    if len(set(x_coords)) > 1 and len(set(y_coords)) > 1:
        # If we have variation in both x and y
        dx = np.gradient(concentrations, x_coords)[-1]  # Most recent gradient
        dy = np.gradient(concentrations, y_coords)[-1]
        print(f"Calculated (x, y) gradients - dx: {dx:.4f}, dy: {dy:.4f}")
        return dx, dy
    elif len(set(x_coords)) > 1:
        # Only x variation
        dx = np.gradient(concentrations, x_coords)[-1]
        return dx, 0
    elif len(set(y_coords)) > 1:
        # Only y variation  
        dy = np.gradient(concentrations, y_coords)[-1]
        return 0, dy
    else:
        return 0, 0
    

# Cast and Surge: Turn Towards Source Once Flow is Detected
def turnToSource(mc, logger):
    global previous_error, integral, previous_time, current_time, drone_heading, flowMap, desired_flow_angle, min_flow_threshold, angle_threshold, Kp, Ki, Kd, last_error
    while (logger.flowAngle >= (desired_flow_angle + angle_threshold)) or (
            logger.flowAngle <= (desired_flow_angle - angle_threshold)):
        # print(f"Flow angle at: {logger.flowAngle}, Flow Magnitude is: {logger.flowMag}")
        flowMap.append((logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))
        error = desired_flow_angle - logger.flowAngle
        current_time = logger.microsecond
        if previous_time is None or previous_time == 0: previous_time = current_time - 0.1
        delta_time = current_time - previous_time
        integral += error * delta_time
        turn_command = int(Kp * error + Ki * integral)
        previous_error = error
        previous_time = current_time

        if abs(error) <= angle_threshold:  # if error is within desired threshold, hover
            print(
                f"Within Threshold. Moving towards detected flow at {logger.flowAngle} degrees with a magnitude of {logger.flowMag}")
            mc.stop()  # Start Non-Blocking Turn
            time.sleep(0.05)
            flowMap.append((logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))

        elif (270 <= logger.flowAngle <= 360) or ((desired_flow_angle + angle_threshold) <= logger.flowAngle <= 270):
            right_command = abs(turn_command) / 100  # Convert to angular rate in degrees/s, scale down
            right_command = max(min(right_command, 1.0), 0)  # Scale to Crazyflie range
            print(
                f"Turning left, source angle & Magnitude: {logger.flowAngle} | {logger.flowMag} ; error: {error} ; command:{right_command}")
            mc.start_turn_left(right_command * 90)  # Convert to appropriate angular rate
            time.sleep(0.05)
            flowMap.append((logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))

        elif (90 <= logger.flowAngle <= (desired_flow_angle - angle_threshold)) or (0 <= logger.flowAngle <= 90):
            left_command = abs(turn_command) / 100  # Convert to angular rate in degrees/s, scale down
            left_command = max(min(left_command, 1.0), 0)  # Scale to Crazyflie range
            left_command = abs(left_command)  # Make positive for turn_left function
            print(
                f"Turning right, source angle & Magnitude: {logger.flowAngle} | {logger.flowMag} ; error:{error} ; command:{left_command}")
            mc.start_turn_right(left_command * 90)  # Convert to appropriate angular rate
            time.sleep(0.05)
            flowMap.append((logger.latest_bx, logger.latest_by, logger.flowMag, logger.flowAngle))

        last_error = error


def spiralMove(mc, logger):
    radius = 0.5  # initial radius
    angle = 0
    angle_increment = 10  # degrees
    radius_increment = 0.1  # meters

    while angle < 720:  # Two full rotations
        # Calculate target position in spiral
        rad_angle = np.radians(angle)
        target_x = radius * np.cos(rad_angle)
        target_y = radius * np.sin(rad_angle)

        # Move to target position
        mc.start_linear_motion(target_x, target_y, 0)
        time.sleep(0.3)  # Adjust time to control speed

        # Update angle and radius for next point in spiral
        angle += angle_increment
        if angle >= 360:
            angle = angle % 360
            radius += radius_increment
        print(f"Spiral moving... Current angle: {angle}°, radius: {radius}m")


def checkRangers(logger):
    print(logger.range_front, logger.range_back, logger.range_left, logger.range_right)
    if logger.range_front < MIN_RANGER_DISTANCE:
        raise InterruptedError("Obstacle detected in front! Stopping forward motion.")
    if logger.range_back < MIN_RANGER_DISTANCE:
        raise InterruptedError("Obstacle detected in back! Stopping backward motion.")
    if logger.range_left < MIN_RANGER_DISTANCE:
        raise InterruptedError("Obstacle detected on right! Stopping leftward motion.")
    if logger.range_right < MIN_RANGER_DISTANCE:
        raise InterruptedError("Obstacle detected on right! Stopping rightward motion.")
    return False



# === MAIN PROGRAM ===
if __name__ == '__main__':
    cflib.crtp.init_drivers(enable_debug_driver=False)
    log_path = get_log_filename()

    with open(log_path, mode="w", newline='') as csv_file:
        fieldnames = [
            "Month", "Day", "Hour", "Minute", "Second", "Microsecond",
            "Speed", "Vx", "Vy", "PosX", "PosY", "Bx", "By",
            "Flow Mag", "Flow Angle", "Gas Concentration", "Range Front", 
            "Range Back", "Range Left", "Range Right",
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

            # Take off
            print("Taking off...")
            mc.take_off(HEIGHT)
            time.sleep(2)

            try:
                # Wait for calibration
                while not logger.calibrated:
                    print("Waiting for calibration...")
                    time.sleep(1)

                print("Calibrated! Starting Navigation Mode...")
                time.sleep(2)

                # ===== CHOOSE NAVIGATION MODE =====

                # Method 1: Direct velocity control (recommended)
                # navigate_to_wind_source(mc, logger)

                # Method 2: Turn and surge (alternative)
                # alternative_turn_and_surge(mc, logger)

                # Method 3: Spiral Test (Spiral Motion Only Test)
                # spiralTest(mc, logger)

                # Method 4: Spiral Search then Approach (Alternative Algorithm)
                # spiralSearch(mc, logger)

                # Method 5: Cast and Surge (Alternative Algorithm)
                # cast_and_surge(mc, logger)

                # Method 6: State Estimator Test (No Wind Navigation)
                # stateEstimatorTest(mc, logger)

                # Method 7: Multiranger Test (Obstacle Avoidance Test)
                # multirangerTest(mc, logger)

                # Method 8: Sensor Response Test (Flow Sensor Response Only Test)
                # sensorResponseTest(mc, logger)

                # Method 9: In-Flight Turn Test (Flow-Based Turning Test)
                # inFlightTurnTest(mc, logger)

                # Method 10: Gradient Gas Detection Test (Gas Gradient Detection Test)
                # gradientGasDetectionTest(mc, logger)

                # Method 11: Gradient Search (Gas Gradient-Based Navigation Test)
                gradientSearch(mc, logger)

            finally:
                print("Stopping logging...")
                logger.running = False
                logger.join()
                mc.stop()
                # time.sleep(1)
                mc.land()
                time.sleep(3)
                print("Sending DISARM request…")
                platform.send_arming_request(False)
                print("Disarmed. Program complete.")
