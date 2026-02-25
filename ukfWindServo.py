import logging
import time
import os
import csv
from datetime import datetime
from threading import Thread
import keyboard
import cflib.crtp
import random
from collections import deque
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.platformservice import PlatformService
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.log import LogConfig
import liveWindUKF
import numpy as np
from liveBouts import initialize_detector, run_bout

# === CONFIGURATION ===
URI = 'radio://0/80/2M/E7E7E7E7E7'
directory_path = r"C:\Users\ltjth\Documents\Research\UKF_Data"
base_filename = "troubleshooting"
file_extension = ".csv"
first_flow_X, first_flow_Y = None, None
last_flow_X, last_flow_Y = None, None
lost_flow_X, lost_flow_Y = None, None

SPEEDS = [0.5]
SIDE_DISTANCE = 3.0
HEIGHT = 1.0

# PID Controller Variables
Kp = 0.6
Ki = 0
Kd = 0.08
integral = 0
previous_error = 0
previous_time = 0
current_time = 0
headings = [0, 90, 180, 270]
drone_heading = 0

# Navigation Vars
min_flow_threshold = 15
angle_threshold = 8
desired_flow_angle = 0
flowMap = []
gas_gradient = []
first_flow_X = 0
first_flow_Y = 0
last_flow_X = 0
last_flow_Y = 0
halfFlag = False
# Wind source navigation parameters
SEARCH_SPEED = 0.3
SEARCH_SPEED_X = 0.1
SEARCH_SPEED_Y = 0.5
SEARCH_DURATION = 0.7
APPROACH_SPEED = 0.3

MAX_FLOW_THRESHOLD = 550
LOCAL_MAX_FLOW = 0
TURN_RATE = 45
minWind = 0.015
absminWind = minWind - 0.02
minGas = 20
maxGas = 32

# Cast and Surge Algorithm Parameters
FORWARD_MOVE_DURATION = 1
CAST_DURATION = 2
CAST_DISTANCE = 1.5
CAST_DISTANCE_X = 0.025
CAST_DISTANCE_Y = 1

STEP_DISTANCE = 0.5
TEST_DISTANCE = 1.5

# Flow thresholds (flow_mag units, not UKF wind)
MIN_FLOW_MAG = 15   # Minimum flow_mag to consider wind present
MAX_FLOW_MAG = 200    # flow_mag threshold to consider source reached

# Multiranger Parameters
MIN_RANGER_DISTANCE = 1600

# Gas Sensor Parameters
GAS_THRESHOLD = 2
MAX_GAS_THRESHOLD = 1023
local_gas_concentration = []
gas_gradients = []

calibrationNumber = 500
gas_slope = 0

global state
#state = 0
# === PLUME DETECTION STATE ===
_plume_state = {
    'in_plume': False,
    'plume_entry_time': None,
    'gas_window': [],
    'bout_amplitude': 0,
    'avg_slope': 0,
    'initialized': False,
    'entry_amp_threshold': 25.0,
    'entry_wind_threshold': minWind,
    'slope_threshold': 0.0,
    'gas_window_size': 20
}


# === FILE HANDLING ===
def get_log_filename():
    file_number = 1
    while True:
        csv_filename = f"{base_filename}{file_number}{file_extension}"
        full_path = os.path.join(directory_path, csv_filename)
        if not os.path.exists(full_path):
            return full_path
        file_number += 1


# === PLUME DETECTION FUNCTIONS ===
def init_plume_detection(fs=100, hl=0.025, entry_amp_threshold=25.0,
                         entry_wind_threshold=0.03, slope_threshold=0.0):
    global _plume_state
    initialize_detector(fs=fs, hl=hl, ampthresh=0)
    _plume_state['entry_amp_threshold'] = entry_amp_threshold
    _plume_state['entry_wind_threshold'] = entry_wind_threshold
    _plume_state['slope_threshold'] = slope_threshold
    _plume_state['gas_window_size'] = 20
    _plume_state['initialized'] = True
    print(f"Plume detector initialized: amp_thresh={entry_amp_threshold}, wind_thresh={entry_wind_threshold}, slope_thresh={slope_threshold}")


def check_plume(gas_reading, wind_magnitude):
    global _plume_state

    if not _plume_state['initialized']:
        raise RuntimeError("Must call init_plume_detection() first!")

    bout_detected, bout_amplitude, bout_frequency = run_bout(gas_reading)
    _plume_state['bout_amplitude'] = bout_amplitude

    _plume_state['gas_window'].append(gas_reading)
    if len(_plume_state['gas_window']) > _plume_state['gas_window_size']:
        _plume_state['gas_window'].pop(0)

    if len(_plume_state['gas_window']) >= 10:
        time_period = (len(_plume_state['gas_window']) - 1) / 100.0
        _plume_state['avg_slope'] = (_plume_state['gas_window'][-1] - _plume_state['gas_window'][0]) / time_period
    else:
        _plume_state['avg_slope'] = 0

    if not _plume_state['in_plume']:
        if bout_amplitude > _plume_state['entry_amp_threshold']:
            _plume_state['in_plume'] = True
            _plume_state['plume_entry_time'] = time.time()
            return "ENTERING_PLUME"
        else:
            return "NOT_IN_PLUME"
    else:
        if _plume_state['avg_slope'] < _plume_state['slope_threshold']:
            _plume_state['in_plume'] = False
            _plume_state['plume_entry_time'] = None
            return "EXITING_PLUME"
        else:
            return "IN_PLUME"


def get_plume_info():
    global _plume_state
    plume_duration = 0
    if _plume_state['in_plume'] and _plume_state['plume_entry_time']:
        plume_duration = time.time() - _plume_state['plume_entry_time']
    return {
        'in_plume': _plume_state['in_plume'],
        'bout_amplitude': _plume_state['bout_amplitude'],
        'avg_slope': _plume_state['avg_slope'],
        'plume_duration': plume_duration
    }


def reset_plume_detection():
    global _plume_state
    _plume_state['in_plume'] = False
    _plume_state['plume_entry_time'] = None
    _plume_state['gas_window'] = []
    _plume_state['avg_slope'] = 0


# === LOGGING THREAD ===
class LoggerThread(Thread):
    def __init__(self, cf, writer):
        super().__init__()
        self.cf = cf
        self.writer = writer
        self.running = True

        self.calib_window = calibrationNumber
        self.bx_window = []
        self.by_window = []
        self.bz_window = []
        self.thresholdGasBuffer = []
        self.averaging_windows = {}
        self.gas_con_window = []
        self.gas_con_window_size = 10
        self.bx_offset = 0
        self.by_offset = 0
        self.gas_con_offset = 0
        self.calibrated = False

        self.month = 0
        self.day = 0
        self.hour = 0
        self.minute = 0
        self.second = 0
        self.microsecond = 0

        self.droneX = 0
        self.droneY = 0
        self.droneVX = 0
        self.droneVY = 0
        self.droneV = 0

        self.latest_bx = 0.0
        self.latest_by = 0.0
        self.flowAngle = 0.0
        self.flow_mag = 0.0
        self.gas_con = 0
        self.elapsed = 0
        self.wind = 0
        self.airflow = 0

        self.range_front = 0.0
        self.range_back = 0.0
        self.range_left = 0.0
        self.range_right = 0.0
        self.gas_cal = 0

        self.plume_status = "NOT_IN_PLUME"

    def moving_average(self, value, window_length, key='default'):
        if key not in self.averaging_windows:
            self.averaging_windows[key] = deque(maxlen=window_length)
        elif self.averaging_windows[key].maxlen != window_length:
            old_data = list(self.averaging_windows[key])
            self.averaging_windows[key] = deque(old_data, maxlen=window_length)
        self.averaging_windows[key].append(value)
        return sum(self.averaging_windows[key]) / len(self.averaging_windows[key])

    def run(self):
        log_conf_1 = LogConfig(name='Flow', period_in_ms=10)
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

        logMulti = LogConfig(name='Rangers', period_in_ms=10)
        logMulti.add_variable('range.front', 'float')
        logMulti.add_variable('range.back', 'float')
        logMulti.add_variable('range.left', 'float')
        logMulti.add_variable('range.right', 'float')

        def log_data_1(timestamp, data, logconf):
            global gas_slope, state
            bx = -data['windSensor.flowX']
            by = -data['windSensor.flowY']
            bz = data['windSensor.flowZ']
            gas = data['windSensor.gas']
            state = 0

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

            vx = data['stateEstimate.vx']
            vy = data['stateEstimate.vy']
            vz = data['stateEstimate.vz']
            px = data['stateEstimate.x']

            bx_cal = bx - self.bx_offset
            by_cal = by - self.by_offset
            bz_cal = bz - self.bz_offset
            gas_cal = gas - self.gas_offset

            bx_cal = self.moving_average(bx_cal, window_length=20, key='bx')
            by_cal = self.moving_average(by_cal, window_length=20, key='by')
            bz_cal = self.moving_average(bz_cal, window_length=20, key='bz')

            angle = np.degrees(np.arctan2(by_cal, bx_cal)) % 360
            flow_mag = np.sqrt(bx_cal ** 2 + by_cal ** 2)

            droneV = np.sqrt(vx ** 2 + vy ** 2)
            airspeed, wind, empirical = liveWindUKF.run_ukf(2 * flow_mag, droneV)

            wind = self.moving_average(wind, window_length=40, key='wind')

            self.flow_mag = flow_mag
            self.wind = wind
            self.airflow = airspeed
            self.flowAngle = angle
            self.gas_con = gas_cal
            self.droneX = px

            # PLUME DETECTION - runs every sample
            self.plume_status = check_plume(gas_cal, wind)

            self.latest_data_1 = {
                "Time": time.time_ns() // 1000, "State": state,
                "Vx": vx, "Vy": vy, "Vz": vz,
                "Bx": bx_cal, "By": by_cal, "Bz": bz_cal,
                "flow_mag": flow_mag, "FlowAngle": angle, "Gas": self.gas_con, "GasRaw": data['windSensor.gas'],
                "BoutAmp": _plume_state['bout_amplitude'],
                "Airspeed": airspeed, "Wind": wind, "PosX": px
            }

        def log_data_2(timestamp, data, logconf):
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

        self.cf.log.add_config(log_conf_1)
        self.cf.log.add_config(log_conf_2)
        self.cf.log.add_config(logMulti)

        log_conf_1.data_received_cb.add_callback(log_data_1)
        log_conf_2.data_received_cb.add_callback(log_data_2)
        logMulti.data_received_cb.add_callback(logMulti_data)

        log_conf_1.error_cb.add_callback(log_error)
        log_conf_2.error_cb.add_callback(log_error)
        logMulti.error_cb.add_callback(log_error)

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


# === NAVIGATION FUNCTIONS ===

def turnToSource(mc, logger,bout_already_detected=False):
    global previous_error, integral, previous_time, current_time, drone_heading, flowMap, \
        desired_flow_angle, min_flow_threshold, angle_threshold, Kp, Ki, Kd, last_error

    bout_detected = bout_already_detected  # seed with prior knowledge
    prev_bout_state = bout_already_detected

    # If we already have a bout, don't require wind above threshold to start turning
    # Wait briefly for wind to register after cast motion stops
    if bout_detected:
        deadline = time.time() + 2.0
        while logger.flow_mag < MIN_FLOW_MAG and time.time() < deadline:
            time.sleep(0.05)
        if logger.flow_mag < MIN_FLOW_MAG:
            print(f"Bout confirmed but wind absent after wait - returning to zigzag")
            return False, True  # bout known but can't orient without wind

    while logger.flow_mag >= 4:
        error = desired_flow_angle - logger.flowAngle
        if error > 180:
            error -= 360
        elif error < -180:
            error += 360

        current_time = logger.microsecond
        if previous_time is None or previous_time == 0:
            previous_time = current_time - 0.1
        delta_time = current_time - previous_time
        integral += error * delta_time
        turn_command = int(Kp * error + Ki * integral)
        previous_error = error
        previous_time = current_time

        # Check for bout during turn - print only on rising edge
        currently_in_plume = logger.plume_status in ["ENTERING_PLUME", "IN_PLUME"]
        if currently_in_plume and not prev_bout_state:
            info = get_plume_info()
            print(f"🟢 BOUT during turn! Amp: {info['bout_amplitude']:.2f} | Gas: {logger.gas_con:.1f}")
            bout_detected = True
        prev_bout_state = currently_in_plume

        if abs(error) <= angle_threshold:
            info = get_plume_info()
            print(f"✅ Aligned | Angle: {logger.flowAngle:.1f}° Mag: {logger.flow_mag:.1f} | Bout: {bout_detected}")
            return True, bout_detected

        elif 180 < logger.flowAngle <= ((desired_flow_angle + 360) - angle_threshold):
            right_command = abs(turn_command) / 100
            right_command = max(min(right_command, 1.0), 0)
            bout_flag = "🟢" if bout_detected else "⚪"
            print(f"RTurn wind: {logger.wind:.3f} err: {abs(error):.1f}° Angle: {logger.flowAngle:.1f}° Mag: {logger.flow_mag:.1f} {bout_flag} {logger.plume_status}")
            mc.start_turn_right(right_command * 90)
            time.sleep(0.01)
            flowMap.append((logger.latest_bx, logger.latest_by, logger.flow_mag, logger.flowAngle))

        elif ((desired_flow_angle + angle_threshold)) <= logger.flowAngle <= 180:
            left_command = abs(turn_command) / 100
            left_command = max(min(left_command, 1.0), 0)
            left_command = abs(left_command)
            bout_flag = "🟢" if bout_detected else "⚪"
            print(f"LTurn wind: {logger.wind:.3f} err: {abs(error):.1f}° Angle: {logger.flowAngle:.1f}° Mag: {logger.flow_mag:.1f} {bout_flag} {logger.plume_status}")
            mc.start_turn_left(left_command * 90)
            time.sleep(0.01)
            flowMap.append((logger.latest_bx, logger.latest_by, logger.flow_mag, logger.flowAngle))

        last_error = error

    # flow_mag dropped below threshold
    # If we already detected a bout, still report aligned=True so approach can run


    if bout_detected:
        print(f"Wind faded during turn but bout was detected")
        return False, bout_detected

    print(f"Lost wind during turn (wind: {logger.wind:.1f})")
    return False, False


k = 0
last_known_direction = 0

def windStateMachine(mc, logger):
    global castFlag, halfFlag
    while True:
        zigZag(mc, logger)
        halfFlag = True

        bout_from_cast = logger.plume_status in ["ENTERING_PLUME", "IN_PLUME"]

        aligned, bout_during_turn = turnToSource(mc, logger, bout_already_detected=bout_from_cast)

        if not aligned:
            if bout_during_turn:
                print("Had bout but lost wind before aligning - re-casting to reacquire")
                castFlag = False  # need to reacquire wind
                continue

        if bout_during_turn or bout_from_cast:
            print("🟢 Bout confirmed - approaching source")
        else:
            print("Aligned with wind, no bout yet - starting zigzag search")
            zigZag(mc, logger)
            halfFlag = True
            if logger.plume_status not in ["ENTERING_PLUME", "IN_PLUME"]:
                print("No plume found during zigzag - returning to search")
                castFlag = False
                reset_plume_detection()
                continue

        reached_source = approach_wind_source(mc, logger)

        if reached_source:
            print("Source reached!")
            return
        else:
            print("Lost plume during approach - returning to search")
            castFlag = False
            reset_plume_detection()
            continue
# def windStateMachine(mc, logger):
#     """
#     Main navigation state machine.
#
#     Flow:
#     1. zigZag → search until wind + bout detected
#     2. turnToSource → align with wind, watching for bouts the whole time
#        - bout_detected flag is sticky (once set, keeps turn going to completion)
#        - If wind fades but bout was already detected, still proceeds to approach
#        - If bout detected during turn → skip zigzag, go straight to approach
#        - If no bout after aligning → zigzag crosswind to find plume
#     3. approach_wind_source → surge upwind with confirmation window for plume loss
#     """
#     while True:
#         # Step 1: Zigzag search until wind + bout detected
#         zigZag(mc, logger)
#
#         # Step 2: Turn to face wind, bout detection sticky throughout
#         aligned, bout_during_turn = turnToSource(mc, logger)
#
#         if not aligned:
#             print("Lost wind during turn (no prior bout) - returning to search")
#             reset_plume_detection()
#             continue
#
#         if bout_during_turn:
#             print("🟢 Bout detected during turn - approaching after this swerve right quick")
#         else:
#             # Aligned with wind but no bout yet - zigzag crosswind to find plume
#             print("Aligned with wind, no bout yet - starting zigzag search")
#             zigZag(mc, logger)
#
#             if logger.plume_status not in ["ENTERING_PLUME", "IN_PLUME"]:
#                 print("No plume found during zigzag - returning to search")
#                 reset_plume_detection()
#                 continue
#
#         # Step 3: Wind + bout confirmed - approach the source
#         reached_source = approach_wind_source(mc, logger)
#
#         if reached_source:
#             print("Source reached!")
#             return
#         else:
#             print("Lost plume during approach - returning to search")
#             reset_plume_detection()
#             continue


def testStraight(mc, logger):
    """Fly forward until wind detected (flow_mag > MIN_FLOW_MAG). Bout NOT required."""
    while True:
        mc.start_linear_motion(0.4, 0, 0)
        time.sleep(0.01)

        info = get_plume_info()
        print(f"⚪ Straight | FlowMag: {logger.flow_mag:.2f} | Gas: {logger.gas_con:.1f} | Bout Amp: {info['bout_amplitude']}")

        if logger.flow_mag > MIN_FLOW_MAG:
            print(f"💨 Wind detected! FlowMag: {logger.flow_mag:.2f}")
            mc.stop()
            return

castFlag = False
CAST_DISTANCE_YL = 0

def zigZag(mc, logger):
    """Cast crosswind looking for a bout. Returns when plume detected or search exhausted."""
    global CAST_DISTANCE_X, CAST_DISTANCE_Y,CAST_DISTANCE_YL ,castFlag, halfFlag
    CAST_DISTANCE_Y = 0.5
    increment = 0.75

    CAST_DISTANCE_YL = CAST_DISTANCE_Y / 2 if not halfFlag else CAST_DISTANCE_Y
    if castFlag and logger.wind < minWind:
        print("Re-entering zigzag after wind loss - holding for reacquisition...")
        deadline = time.time() + 3.0
        while logger.wind < minWind and time.time() < deadline:
            mc.stop()
            time.sleep(0.1)
        if logger.wind < minWind:
            print("Wind did not return - resetting castFlag")
            castFlag = False
    while logger.plume_status == "NOT_IN_PLUME":
        # if wind but plume_status = not in plume, set flag to initial cast
        # then after flag is set to initial cast, we can only leave zigzag after being in plume
        # ========== LEFT CAST ==========
        start_posY = logger.droneY
        mc.start_linear_motion(SEARCH_SPEED_X, SEARCH_SPEED_Y, 0)

        while abs(start_posY - logger.droneY) < CAST_DISTANCE_YL:
            if logger.plume_status in ["ENTERING_PLUME", "IN_PLUME"] and castFlag == True:
                info = get_plume_info()
                print(f"🟢 Plume detected during left cast! Amp: {info['bout_amplitude']:.2f}")
                #mc.stop()
                return
            # Replace with wind ukf
            #if logger.flow_mag > MIN_FLOW_MAG and castFlag == False:
            if logger.wind > minWind and castFlag == False:
                print(f"WIND detected during left cast! Mag: {logger.flow_mag:.2f}")
                castFlag = True
                #mc.stop()
                return
            info = get_plume_info()
            print(f"CL Wind: {logger.wind:.3f} | Gas: {logger.gas_con:.1f} | FlowMag: {logger.flow_mag:.1f} | Amp: {info['bout_amplitude']}")
            ranger_status = checkRangers(logger)

            if ranger_status == 1:
                print("obstacle on left")
                CAST_DISTANCE_Y = 0.5
                break



            time.sleep(0.01)
        mc.stop()
        time.sleep(0.5)

        if logger.plume_status in ["ENTERING_PLUME", "IN_PLUME"]and castFlag == True:
            return

        # ========== RIGHT CAST ==========

        start_posY = logger.droneY
        mc.start_linear_motion(SEARCH_SPEED_X, -SEARCH_SPEED_Y, 0)
        CAST_DISTANCE_Y += increment

        while abs(start_posY - logger.droneY) < CAST_DISTANCE_Y:
            if logger.plume_status in ["ENTERING_PLUME", "IN_PLUME"]and castFlag == True:
                info = get_plume_info()
                print(f"🟢 Plume detected during right cast! Amp: {info['bout_amplitude']:.2f}")
                mc.stop()
                return
            # Replace with wind ukf
            if logger.wind > minWind and castFlag == False:
                print(f"WIND detected during right cast! Mag: {logger.flow_mag:.2f}")
                castFlag = True
                # mc.stop()
                return
            info = get_plume_info()
            print(f"CR Wind: {logger.wind:.3f}| Gas: {logger.gas_con:.1f} | FlowMag: {logger.flow_mag:.1f} | Amp: {info['bout_amplitude']}")
            ranger_status = checkRangers(logger)

            if ranger_status == 2:
                print("obstacle on right")
                CAST_DISTANCE_Y = 0.5
                break



            time.sleep(0.01)
        mc.stop()
        time.sleep(0.5)

        CAST_DISTANCE_Y += increment
        CAST_DISTANCE_YL = CAST_DISTANCE_Y  # now safe — first_pass already done


        if logger.plume_status in ["ENTERING_PLUME", "IN_PLUME"]:
            return

def turnAwayWall(mc, logger, castDir):
    if castDir == 'R':
        print("Turning left away from the wall")
        mc.stop()
        time.sleep(0.5)
        mc.turn_left(90)
        time.sleep(0.5)
    if castDir == 'L':
        print("Turning right away from the wall")
        mc.stop()
        time.sleep(0.5)
        mc.turn_right(90)
        time.sleep(0.5)


def checkRangers(logger):
    # if logger.range_front < MIN_RANGER_DISTANCE:
    #     raise InterruptedError("Obstacle detected in front! Stopping front motion.")
    if logger.range_back < 300:
        raise InterruptedError("Obstacle detected in back! Stopping backward motion.")
    if logger.range_left < MIN_RANGER_DISTANCE:
        return 1
    if logger.range_right < MIN_RANGER_DISTANCE:
        return 2
    return 0


windBuffer = []
gasBuffer = []


def approach_wind_source(mc, logger):
    """
    Surge upwind while plume is detected.
    Requires PLUME_EXIT_CONFIRM consecutive EXITING/NOT samples before giving up,
    to avoid bailing on momentary turbulence dips.
    Returns True if source reached, False if plume confirmed lost.
    """
    global flowMap, last_flow_X, last_flow_Y, minWind
    PLUME_EXIT_CONFIRM = 10  # consecutive non-plume samples needed to exit

    info = get_plume_info()
    print(f"Approaching wind source.. Wind: {logger.wind:.3f} FlowMag: {logger.flow_mag:.1f} | Amp: {info['bout_amplitude']:.2f}")
    windBuffer.clear()
    gasBuffer.clear()
    exit_counter = 0

    while logger.wind >= 0.01 or logger.flow_mag >= 9:
        # windBuffer.append(logger.flow_mag)``
        gasBuffer.append(logger.gas_con)
        # if len(windBuffer) > 3:
        #     windBuffer.pop(0)
        if len(gasBuffer) > 3:
            gasBuffer.pop(0)

        avgGas = sum(gasBuffer) / len(gasBuffer)
        # Source reached
        source_angle = logger.flowAngle
        vx = APPROACH_SPEED * np.cos(np.radians(source_angle))
        vy = APPROACH_SPEED * np.sin(np.radians(source_angle))

        info = get_plume_info()
        exit_str = f" [exit:{exit_counter}/{PLUME_EXIT_CONFIRM}]" if exit_counter > 0 else ""
        print(
            f"🟢 Approaching  Wind: {logger.wind:.3f}| Gas: {logger.gas_con:.1f} | FlowMag: {logger.flow_mag:.1f}  | Slope: {info['avg_slope']:.2f}{exit_str}")

        mc.start_linear_motion(vx, vy, 0)
        time.sleep(0.05)

        if logger.range_front < MIN_RANGER_DISTANCE:
            print("Source reached hoorah")
            mc.stop()
            mc.land()
            return True

        # Plume exit: require PLUME_EXIT_CONFIRM consecutive bad samples
        if info['avg_slope'] < _plume_state['slope_threshold']:
            exit_counter += 1
            if exit_counter >= PLUME_EXIT_CONFIRM:
                print(
                    f"🔴 Lost plume during approach ({PLUME_EXIT_CONFIRM} consecutive) | Slope: {info['avg_slope']:.2f}")
                #mc.stop()
                return False
        else:
            exit_counter = 0


    # Flow dropped below threshold
    print(f"Lost wind during approach (wind: {logger.wind:.3f})")
    #mc.stop()
    return False


# === MAIN PROGRAM ===
if __name__ == '__main__':
    cflib.crtp.init_drivers(enable_debug_driver=False)
    log_path = get_log_filename()

    with open(log_path, mode="w", newline='') as csv_file:
        fieldnames = [
            "Time", "State",
            "Vx", "Vy", "Vz",
            "Bx", "By", "Bz", "flow_mag", "FlowAngle", "Gas","GasRaw", "BoutAmp", "Airspeed", "Wind",
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

            init_plume_detection(
                fs=100,
                hl=0.025,
                entry_amp_threshold=25,
                entry_wind_threshold=minWind,
                slope_threshold=0.0
            )

            mc = MotionCommander(scf, default_height=HEIGHT)

            print("Taking off...")
            mc.take_off(HEIGHT)

            logger = LoggerThread(cf, writer)
            logger.start()

            time.sleep(calibrationNumber / 100)

            try:
                windStateMachine(mc, logger)

            finally:
                print("Stopping logging...")
                mc.land()
                time.sleep(3)
                logger.running = False
                logger.join()

                print("Sending DISARM request…")
                platform.send_arming_request(False)
                print("Disarmed. Program complete.")
