#GAS_THRESHOLD = 
# Wind source navigation parameters
# SEARCH_SPEED = 0.3
# SEARCH_SPEED_X = 0.3
# SEARCH_SPEED_Y = 0.3
# SEARCH_DURATION = 0.7  # seconds per search segment
# APPROACH_SPEED = 0.5
# APPROACH_SPEED_X = 0.5
# APPROACH_SPEED_Y = 1
# MAX_FLOW_THRESHOLD = 550  # When to consider we've reached the source
# LOCAL_MAX_FLOW = 0  # Parameter used for Cast and Surge Algorithm
# TURN_RATE = 45  # degrees/second for turning towards source

# # Cast and Surge Algorithm Parameters
# FORWARD_MOVE_DURATION = 1  # seconds
# CAST_DURATION = 2  # seconds
# CAST_DISTANCE = 1.5  # meters for unilateral cast
# CAST_DISTANCE_X = 0.1
# CAST_DISTANCE_Y = 0.3

# #random walk
# STEP_DISTANCE = 0.5

# # Test Parameters
# TEST_DISTANCE = 1.5  # meters for state estimator test

# # Multiranger Parameters
# MIN_RANGER_DISTANCE = 1000
# # mm, minimum distance to obstacle

# # Gas Sensor Parameters
# GAS_THRESHOLD = 3  # Threshold for gas concentration
# MAX_GAS_THRESHOLD = 1023  # Threshold for gas concentration to consider source reached
# local_gas_concentration = []
# gas_gradients = []

def uni(mc, logger):
    global flowMap, CAST_DISTANCE_Y
    """
    Approach phase - navigate towards wind source using flow direction
    """
    while True:
        while (logger.gas_con < GAS_THRESHOLD):
            checkRangers(logger)
            start_posY = logger.droneY
            print(logger.gas_con)
            mc.start_linear_motion(0, SEARCH_SPEED_Y, 0)
            while (abs(start_posY - logger.droneY) < CAST_DISTANCE_Y) and (logger.gas_con < GAS_THRESHOLD):
                checkRangers(logger)
                print(logger.gas_con)
                print("Casting Left...")
            mc.stop()
            time.sleep(0.2)

            print(logger.gas_con)
            start_posY = logger.droneY
            mc.start_linear_motion(0, -SEARCH_SPEED_Y, 0)
            while (abs(start_posY - logger.droneY) < CAST_DISTANCE_Y) and (logger.gas_con < GAS_THRESHOLD):
                checkRangers(logger)
                print(logger.gas_con)
                print("Casting Right...")
            mc.stop()
            time.sleep(0.2)

            CAST_DISTANCE_Y += 0.1
        print("Gas Threshold Crossed")
        print(logger.gas_con)
        start_posX = logger.droneX
        mc.start_linear_motion(0.4, 0, 0)
        while (abs(start_posX - logger.droneX) < STEP_DISTANCE) and (logger.gas_con > GAS_THRESHOLD):
            checkRangers(logger)
            print(logger.gas_con)
            print("Surging Forward")
    print("max flow reached, landing...")
    mc.land()
