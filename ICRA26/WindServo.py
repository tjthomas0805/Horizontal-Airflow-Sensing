import logging
import time
import os
import csv
import numpy as np
from threading import Thread
import cflib.crtp
from collections import deque
# from CFLogger import *
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.platformservice import PlatformService
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.log import LogConfig


import liveWindUKF
from liveBouts import initialize_detector, run_bout


# from sklearn.linear_model import LinearRegression

# === CONFIGURATION ===``
URI = 'radio://0/80/2M/E7E7E7E7E7'  # <-- change if needed
directory_path = r"C:\Users\ltjth\Documents\Research\UKF_Data"
base_filename = "noGasnoBreaks_forcalibration"
file_extension = ".csv"
first_flow_X, first_flow_Y = None, None
last_flow_X, last_flow_Y = None, None
lost_flow_X, lost_flow_Y = None, None

# Speeds for square flights (m/s)
SPEEDS = [0.5]
SIDE_DISTANCE = 3.0  # meters
HEIGHT = 0.75  # takeoff height

# PID Controller Variables
Kp = 0.6
Ki = 0
Kd = 0.08
integral = 0
previous_error = 0
previous_time = 0
current_time = 0
headings = [0, 90, 180, 270]
# Drone Variables
drone_heading = 0

# Navigation Vars
min_flow_threshold = 15
angle_threshold = 10
desired_flow_angle = 180
flowMap = []
gas_gradient = []
first_flow_X = 0
first_flow_Y = 0
last_flow_X = 0
last_flow_Y = 0
# once you detect flow add these as a tuple to flowMap

# Wind source navigation parameters
SEARCH_SPEED = 0.3
SEARCH_SPEED_X = 0.1
SEARCH_SPEED_Y = 0.5
SEARCH_DURATION = 0.7  # seconds per search segment
APPROACH_SPEED = 0.5

MAX_FLOW_THRESHOLD = 550  # When to consider we've reached the source
LOCAL_MAX_FLOW = 0  # Parameter used for Cast and Surge Algorithm
TURN_RATE = 45  # degrees/second for turning towards source
minWind = 0.08
absminWind = minWind-0.02
minGas = 20
maxGas = 32
# Cast and Surge Algorithm Parameters
FORWARD_MOVE_DURATION = 1  # seconds
CAST_DURATION = 2  # seconds
CAST_DISTANCE = 1.5  # meters for unilateral cast
CAST_DISTANCE_X = 0.025
CAST_DISTANCE_Y = 1

# random walk
STEP_DISTANCE = 0.5

# Test Parameters
TEST_DISTANCE = 1.5  # meters for state estimator test

# Multiranger Parameters
MIN_RANGER_DISTANCE = 1000
# mm, minimum distance to obstacle

# Gas Sensor Parameters
GAS_THRESHOLD = 2  # Threshold for gas concentration
MAX_GAS_THRESHOLD = 1023  # Threshold for gas concentration to consider source reached
local_gas_concentration = []
gas_gradients = []

calibrationNumber = 500
gas_slope = 0

boutAmpThresh = 0.2
windBoolean = False

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
        # self.angle_window = []
        # self.angle_window_size = 50  # Reduced for faster response

        # Calibration parameters
        self.calib_window = calibrationNumber  # 10 seconds of calibration data
        self.bx_window = []
        self.by_window = []
        self.bz_window = []
        self.thresholdGasBuffer = []
        # Moving average windows - dictionary to store windows for different variables
        self.averaging_windows = {}
        self.gas_con_window = []
        # self.mag_con_window = []
        # self.mag_con_window_size = 20
        self.gas_con_window_size = 10  # Moving average window for gas concentration
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
        self.droneV = 0

        # Flow variables
        self.latest_bx = 0.0
        self.latest_by = 0.0
        self.flowAngle = 0.0
        self.flow_mag = 0.0
        self.gas_con = 0
        self.elapsed = 0
        self.wind = 0
        self.airflow = 0

        # Drone Multiranger variables
        self.range_front = 0.0
        self.range_back = 0.0
        self.range_left = 0.0
        self.range_right = 0.0

        # Gas Bout Stuff
        self.bout_detected = False
        self.bout_amplitude = 0
        self.bout_frequency = 0
        initialize_detector(fs=100, ampthresh=10.0)

    def moving_average(self, value, window_length, key='default'):
        """
        Apply moving average filter with variable window length.

        Args:
            value: The new value to add to the filter
            window_length: The length of the moving average window
            key: A unique identifier for this particular signal (e.g., 'bx', 'gas', 'flow_mag')

        Returns:
            The filtered (averaged) value
        """
        # Create a new deque for this key if it doesn't exist or window size changed
        if key not in self.averaging_windows:
            self.averaging_windows[key] = deque(maxlen=window_length)
        elif self.averaging_windows[key].maxlen != window_length:
            # Window size changed, create new deque with existing data
            old_data = list(self.averaging_windows[key])
            self.averaging_windows[key] = deque(old_data, maxlen=window_length)

        self.averaging_windows[key].append(value)
        return sum(self.averaging_windows[key]) / len(self.averaging_windows[key])

    def run(self):
        log_conf_1 = LogConfig(name='Flow', period_in_ms=10)  # 100 Hz
        log_conf_1.add_variable('stateEstimate.vx', 'float')
        log_conf_1.add_variable('stateEstimate.vy', 'float')
        log_conf_1.add_variable('stateEstimate.vz', 'float')
        log_conf_1.add_variable('windSensor.flowX', 'int16_t')
        log_conf_1.add_variable('windSensor.flowY', 'int16_t')
        log_conf_1.add_variable('windSensor.flowZ', 'int16_t')
        log_conf_1.add_variable('windSensor.gas', 'int16_t')
        log_conf_1.add_variable('stateEstimate.x', 'float')

        log_conf_2 = LogConfig(name='Attitude', period_in_ms=10)
        log_conf_2.add_variable('stateEstimate.y', 'float')
        log_conf_2.add_variable('stateEstimate.z', 'float')
        log_conf_2.add_variable('stateEstimate.qx', 'float')
        log_conf_2.add_variable('stateEstimate.qy', 'float')
        log_conf_2.add_variable('stateEstimate.qz', 'float')
        log_conf_2.add_variable('stateEstimateZ.ratePitch', 'int16_t')
        log_conf_2.add_variable('stateEstimateZ.rateRoll', 'int16_t')
        log_conf_2.add_variable('stateEstimateZ.rateYaw', 'int16_t')

        #
        logMulti = LogConfig(name='Rangers', period_in_ms=10)
        logMulti.add_variable('range.front', 'float')
        logMulti.add_variable('range.back', 'float')
        logMulti.add_variable('range.left', 'float')
        logMulti.add_variable('range.right', 'float')

        def log_data_1(timestamp, data, logconf):
            global gas_slope
            bx = -data['windSensor.flowX']
            by = -data['windSensor.flowY']
            bz = data['windSensor.flowZ']
            gas = data['windSensor.gas']

            # Calibration
            if not self.calibrated:
                print(f"Logger started, please wait {self.calib_window / 100} seconds")
                if len(self.bx_window) < self.calib_window:
                    self.bx_window.append(bx)
                    self.by_window.append(by)
                    self.bz_window.append(bz)
                    self.gas_con_window.append(gas)
                    return
                else:
                    self.bx_offset = np.median(self.bx_window)
                    self.by_offset = np.median(self.by_window)
                    self.bz_offset = np.median(self.bz_window)
                    self.gas_offset = np.median(self.gas_con_window)
                    self.calibrated = True
                    print(f"Calibration done: Bx offset={self.bx_offset:.2f}, By offset={self.by_offset:.2f}")

            # Timestamp

            # State values
            vx = data['stateEstimate.vx']
            vy = data['stateEstimate.vy']
            vz = data['stateEstimate.vz']
            px = data['stateEstimate.x']

            # Calibrated whisker values
            bx_cal = bx - self.bx_offset
            by_cal = by - self.by_offset
            bz_cal = bz - self.bz_offset
            gas_cal = gas - self.gas_offset

            # Apply moving average with different window lengths for each signal
            # Shorter windows = less filtering, faster response
            # Longer windows = more filtering, smoother but slower response
            bx_cal = self.moving_average(bx_cal, window_length=20, key='bx')
            by_cal = self.moving_average(by_cal, window_length=20, key='by')
            bz_cal = self.moving_average(bz_cal, window_length=20, key='bz')
            gas_cal = self.moving_average(gas_cal, window_length=10, key='gas')  # More filtering for gas

            self.thresholdGasBuffer.append(logger.gas_con)

            # Derived quantities
            angle = np.degrees(np.arctan2(by_cal, bx_cal)) % 360
            flow_mag = np.sqrt(bx_cal ** 2 + by_cal ** 2)

            # Less filtering for flow magnitude - faster response
            droneV = np.sqrt(vx ** 2 + vy ** 2)
            airspeed, wind = liveWindUKF.run_ukf(2 * flow_mag, droneV)

            # Bout Detector
            self.bout_detected, self.bout_amplitude, self.bout_frequency = run_bout(gas)

            # Store most recent for merged CSV write

            self.flow_mag = flow_mag
            self.wind = wind
            self.airflow = airspeed
            self.flowAngle = angle
            self.gas_con = gas_cal
            self.droneX = px

            self.latest_data_1 = {
                "Time": time.time_ns() // 1000,
                "Vx": vx, "Vy": vy, "Vz": vz,
                "Bx": bx_cal, "By": by_cal, "Bz": bz_cal,
                "flow_mag": flow_mag, "FlowAngle": angle, "Gas": data['windSensor.gas'],
                "Airspeed": airspeed, "Wind": wind, "PosX": px
            }

        def log_data_2(timestamp, data, logconf):
            # Store attitude and range data
            self.droneY = data['stateEstimate.y']
            self.latest_data_2 = {
                "PosY": data['stateEstimate.y'],
                "PosZ": data['stateEstimate.z'],
                "Qx": data['stateEstimate.qx'],
                "Qy": data['stateEstimate.qy'],
                "Qz": data['stateEstimate.qz'],
                "PitchRate": data['stateEstimateZ.ratePitch'],
                "RollRate": data['stateEstimateZ.rateRoll'],
                "YawRate": data['stateEstimateZ.rateYaw'],

            }

            # Merge and write to CSV if both data sets are available
            if hasattr(self, "latest_data_1") and self.running:
                row = {**self.latest_data_1, **self.latest_data_2}
                self.writer.writerow(row)

        def logMulti_data(timestamp, data, logconf):
            self.range_front = data['range.front']
            self.range_back = data['range.back']
            self.range_left = data['range.left']
            self.range_right = data['range.right']

        def log_error(logconf, msg):
            print(f"Logging error in {logconf.name}: {msg}")

        # -------------------
        # REGISTER LOG CONFIGS
        # -------------------
        self.cf.log.add_config(log_conf_1)
        self.cf.log.add_config(log_conf_2)
        self.cf.log.add_config(logMulti)

        log_conf_1.data_received_cb.add_callback(log_data_1)
        log_conf_2.data_received_cb.add_callback(log_data_2)
        logMulti.data_received_cb.add_callback(logMulti_data)

        log_conf_1.error_cb.add_callback(log_error)
        log_conf_2.error_cb.add_callback(log_error)
        logMulti.error_cb.add_callback(log_error)

        # -------------------
        # START LOGGING
        # -------------------
        try:
            log_conf_1.start()
            log_conf_2.start()
            logMulti.start()

            while self.running:
                time.sleep(0.1)

            log_conf_1.stop()
            log_conf_2.stop()
            logMulti.stop()
        except Exception as e:
            print("Logging error:", e)


def ukfTest(mc, logger):
    global minWind
    # to avoid spike from moving
    # mc.start_fo(0.5)
    mc.start_forward(0.5)
    time.sleep(2)
    while True:
        mc.start_forward(0.5)
        # mc.start_linear_motion(0,0,0,0)
        time.sleep(0.01)
        print(f"Flow Mag: {logger.flow_mag} mT | Wind: {logger.wind} m/s")
        # maybe change to be a window#
        if logger.wind > minWind:
            print("---------------------------WIND-----------------------------------------")
            mc.stop()
            time.sleep(0.2)
            approach_wind_source(mc, logger)
            # turnToSource(mc,logger)
            # mc.stop()
            # mc.land()
            # break


def turnToSource(mc, logger):
    global previous_error, integral, previous_time, current_time, drone_heading, flowMap, desired_flow_angle, min_flow_threshold, angle_threshold, Kp, Ki, Kd, last_error
    while True:
    # while (logger.flowAngle >= (desired_flow_angle + angle_threshold)) or (
    #         logger.flowAngle <= (desired_flow_angle - angle_threshold)):
        # print(f"Flow angle at: {logger.flowAngle}, Flow Magnitude is: {logger.flow_mag}")
        flowMap.append((logger.latest_bx, logger.latest_by, logger.flow_mag, logger.flowAngle))
        error = desired_flow_angle - logger.flowAngle
        current_time = logger.microsecond
        if previous_time is None or previous_time == 0: previous_time = current_time - 0.1
        delta_time = current_time - previous_time
        integral += error * delta_time
        turn_command = int(Kp * error + Ki * integral)
        previous_error = error
        previous_time = current_time

        if abs(error) <= 5:  # if error is within desired threshold, hover
            print(
                f"Within Threshold. Moving towards detected flow at {logger.flowAngle} degrees with a magnitude of {logger.flow_mag}")
            mc.stop()  # Start Non-Blocking Turn
            time.sleep(0.05)
            approach_wind_source(mc, logger)
            return
            # flowMap.append((logger.latest_bx, logger.latest_by, logger.flow_mag, logger.flowAngle))

        elif (270 <= logger.flowAngle <= 360) or ((desired_flow_angle + angle_threshold) <= logger.flowAngle <= 270):
            right_command = abs(turn_command) / 100  # Convert to angular rate in degrees/s, scale down
            right_command = max(min(right_command, 1.0), 0)  # Scale to Crazyflie range
            print(f"Turning Left at {right_command:2f} toward {desired_flow_angle:2f}")
            mc.start_turn_left(right_command * 90)  # Convert to appropriate angular rate
            time.sleep(0.01)
            flowMap.append((logger.latest_bx, logger.latest_by, logger.flow_mag, logger.flowAngle))

        elif (90 <= logger.flowAngle <= (desired_flow_angle - angle_threshold)) or (0 <= logger.flowAngle <= 90):
            left_command = abs(turn_command) / 100  # Convert to angular rate in degrees/s, scale down
            left_command = max(min(left_command, 1.0), 0)  # Scale to Crazyflie range
            left_command = abs(left_command)  # Make positive for turn_left function
            print(f"Turning right at {left_command:2f} toward {desired_flow_angle:2f}")
            mc.start_turn_right(left_command * 90)  # Convert to appropriate angular rate
            time.sleep(0.01)
            flowMap.append((logger.latest_bx, logger.latest_by, logger.flow_mag, logger.flowAngle))

        last_error = error




k = 0
last_known_direction = 0


def local_search(mc, logger, duration):
    global last_known_direction, last_known_position, k
    """
    Cast perpendicular to flow based on which side of plume we drifted to
    """
    print(f"Lost signal - determining which side of plume...")

    # Determine spiral direction based on last known flow
    if last_known_direction <= 90 or last_known_direction >= 270:
        k = -1
        print(f"Drifted LEFT of plume, spiral RIGHT (k={k})")
    else:
        k = 1
        print(f"Drifted RIGHT of plume, spiral LEFT (k={k})")

    # Spiral parameters
    radius = 0.3
    angle = 0
    angle_increment = 10  # degrees
    radius_increment = 0.1

    start_time = time.time()

    while logger.gasSlope < -0.2 and logger.wind < minWind:
        elapsed = time.time() - start_time

        if elapsed >= 15:
            print(f"Local search timeout after {elapsed:.1f}s - signal not reacquired")
            mc.stop()
            return False

        # RECALCULATE target position every iteration!
        rad_angle = np.radians(angle)
        target_x = radius * np.cos(rad_angle)
        target_y = radius * np.sin(rad_angle)

        # Apply the spiral motion with directional bias
        mc.start_linear_motion(target_x, k * target_y, 0)

        print(f"Spiral: r={radius:.2f} θ={angle}° | Wind: {logger.gas_con:.2f} | Elapsed: {elapsed:.1f}s")
        time.sleep(0.15)

        # Update angle and radius for next iteration
        angle += angle_increment
        if angle >= 360:
            angle = angle % 360
            radius += radius_increment

    print(f"Signal reacquired! Wind: {logger.gas_con:.2f}")
    mc.stop()
    return True
def windStateMachine(mc, logger):
    global minWind

    while True:
        if logger.wind > minWind:
            logger.windBoolean = True
        # Search until wind found
        zigZag(mc, logger)

        # Now we have wind signal - keep approaching until we reach source or lose signal
        while True:
            turnToSource(mc, logger)
            # Approach returns True if reached source, False if lost wind
            reached_source = approach_wind_source(mc, logger)

            if reached_source:
                print("Source reached!")
                return

            # Lost signal - try local search
            wind_reacquired = local_search(mc, logger, duration=11)

            if wind_reacquired:
                print("Wind reacquired! Continuing approach...")
                # Inner loop continues, calls approach again
            else:
                print("Local search failed, returning to full zigzag search...")
                break  # Exit inner loop, go back to zigzag

def zigZag(mc, logger):
    global CAST_DISTANCE_X, CAST_DISTANCE_Y, first_flow_X, first_flow_Y, minWind
    """
    Search phase - move forward slowly until wind is detected
    """


    # Keep AND logic - continue while BOTH are below threshold
    CAST_DISTANCE_Y = 0.5
    #First encounter with wind regardless of gas and wind
        #Turn toward wind source
    #If wind AND gas
        #approach
    #If ONLY wind reorient
    
    while logger.wind < minWind:

        start_posX = logger.droneX
        start_posY = logger.droneY
        mc.start_linear_motion(SEARCH_SPEED_X, SEARCH_SPEED_Y, 0)

        while (abs(start_posY - logger.droneY) < CAST_DISTANCE_Y):
            # Check thresholds during casting - exit only if BOTH exceed threshold
            if (logger.wind > minWind) and logger.bout_amplitude > boutAmpThresh:
                print(f"Both thresholds crossed during left cast! slope({logger.gasSlope:.2f}")
                mc.stop()
                return
            print(f"Casting Left  Gas: {logger.gas_con}  Wind: {logger.wind:.2f}")
            time.sleep(0.05)

        mc.stop()
        time.sleep(0.05)
        if (logger.wind > minWind) and (logger.gasSlope > 0.2):  # and (abs(logger.gas_con) > GAS_THRESHOLD):
            print(f"Wind")
            return

        # Cast Right - check thresholds during movement
        start_posX = logger.droneX
        start_posY = logger.droneY
        mc.start_linear_motion(SEARCH_SPEED_X, -SEARCH_SPEED_Y, 0)
        CAST_DISTANCE_Y += 0.3
        while (abs(start_posY - logger.droneY) < CAST_DISTANCE_Y):
            # Check thresholds during casting - exit only if BOTH exceed threshold
            if (logger.wind > minWind) and logger.bout_amplitude > boutAmpThresh:
                print(f"Both thresholds crossed during right cast! slope({logger.gasSlope:.2f}")
                mc.stop()
                return
            print(f"Casting Right slope: {logger.gasSlope} Gas: {logger.gas_con} Wind: {logger.wind:.2f}")
            time.sleep(0.05)

        mc.stop()
        time.sleep(0.05)

        # Expand search area for next iteration
        # CAST_DISTANCE_X += 0.3
        CAST_DISTANCE_Y += 0.3


magBuffer = []
gasBuffer = []
rangeThreshold = 3


def approach_wind_source(mc, logger):
    global flowMap, last_flow_X, last_flow_Y, minWind, last_known_direction, last_known_position
    print("Approaching wind source...")

    magBuffer.clear()
    gasBuffer.clear()

    while logger.wind > minWind:
        magBuffer.append(logger.wind)
        gasBuffer.append(logger.gas_con)
        if len(magBuffer) > 3:
            magBuffer.pop(0)
        if len(gasBuffer) > 3:
            gasBuffer.pop(0)

        avgGas = sum(gasBuffer) / len(gasBuffer)
        avgWind = sum(magBuffer) / len(magBuffer)

        if avgGas >= maxGas:
            print("Source reached hoorah")
            mc.stop()
            mc.land()
            return True

        # Update last known good state
        if avgWind > minWind:
            last_known_direction = logger.flowAngle
            last_known_position = (logger.droneX, logger.droneY)

        # Check for lost signal with hysteresis
        if logger.gasSlope < 0:

            print(f"lost gas slope: {logger.gasSlope}")
            #mc.stop()
            return False

            # Navigate towards source
        source_angle = (logger.flowAngle + 180) % 360
        vx = APPROACH_SPEED * np.cos(np.radians(source_angle))
        vy = APPROACH_SPEED * np.sin(np.radians(source_angle))

        print(
            f"Approaching slope: {logger.gasSlope} Gas: {logger.gas_con} Mag: {logger.flow_mag} Wind: {logger.wind:.2f} @ {logger.flowAngle:.1f}°")

        mc.start_linear_motion(vx, vy, 0)
        time.sleep(0.05)

# def turnTester(mc,logger):
#
#     if logger.wind > minWind:
#         turnToSource(mc,logger)
#     else:
#         mc.start_forward(0.4)

# === MAIN PROGRAM ===
if __name__ == '__main__':
    cflib.crtp.init_drivers(enable_debug_driver=False)
    log_path = get_log_filename()

    with open(log_path, mode="w", newline='') as csv_file:
        fieldnames = [
            "Time",
            "Vx", "Vy", "Vz",
            "Bx", "By", "Bz", "flow_mag", "FlowAngle", "Gas", "Airspeed", "Wind",
            "PosX", "PosY", "PosZ",
            "Qx", "Qy", "Qz",
            "PitchRate", "RollRate", "YawRate"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            cf = scf.cf
            platform = PlatformService(crazyflie=cf)

            print("Connected — sending ARM request…")
            platform.send_arming_request(True)
            print("Armed! (but not taking off)")

            # Initialize motion commander
            mc = MotionCommander(scf, default_height=HEIGHT)

            # Take off
            print("Taking off...")
            mc.take_off(HEIGHT)
            # time.sleep(2)
            # Start logging thread
            logger = LoggerThread(cf, writer)
            logger.start()

            time.sleep(calibrationNumber / 100)

            try:

                windStateMachine(mc, logger)
                # print("Turn on fan")
                # time.sleep(6)
                # turnToSource(mc,logger)


            finally:
                print("Stopping logging...")
                mc.land()
                time.sleep(3)
                logger.running = False
                logger.join()

                print("Sending DISARM request…")
                platform.send_arming_request(False)
                print("Disarmed. Program complete.")
# === NAVIGTION MODES ===
def coicle(mc, logger):
    print("Searching for wind with spiral cast...")
    radius = 0.3  # initial radius
    angle = 0
    angle_increment = 10  # degrees
    radius_increment = 0.1  # meters
    startTime = time.time()

    while True:
        elapsedTime = time.time() - startTime
        # Calculate target position in spiral
        if elapsedTime >= 30:
            print("Local search timeout")
            return False
        rad_angle = np.radians(angle)
        target_x = radius * np.cos(rad_angle)
        target_y = radius * np.sin(rad_angle)
        print(f"Spiralinggggg weeeeee x: {target_x:.2f} y:{target_y:.2f} {elapsedTime:.2f}")
        # Move to target position
        mc.start_linear_motion(target_x, target_y, 0)
        time.sleep(0.15)  # Adjust time to control speed`````

        # Update angle and radius for next point in spiral
        angle += angle_increment
        print(radius)
        if angle >= 360:
            angle = angle % 360
            radius += radius_increment
        # print(f"No wind detected (mag: {logger.wind:.2f}), continuing spiral search...")
