# For desired angle of 0 with threshold of 5
# If magnet is flipped, change desired to 180 and change if statement logic

def turnToSource(mc, logger):
    global previous_error, integral, previous_time, current_time, drone_heading, flowMap, desired_flow_angle, min_flow_threshold, angle_threshold, Kp, Ki, Kd, last_error
    while True:#(logger.flowAngle >= (desired_flow_angle + angle_threshold)) or (logger.flowAngle <= (desired_flow_angle - angle_threshold)):
        # print(f"Flow angle at: {logger.flowAngle}, Flow Magnitude is: {logger.flow_mag}")
        #flowMap.append((logger.latest_bx, logger.latest_by, logger.flow_mag, logger.flowAngle))
        #sourceAngle = (logger.flowAngle + 180)%360
        error = desired_flow_angle - logger.flowAngle

        if error > 180:
            error -= 360
        elif error < -180:
            error += 360
        current_time = logger.microsecond
        if previous_time is None or previous_time == 0: previous_time = current_time - 0.1
        delta_time = current_time - previous_time
        integral += error * delta_time
        turn_command = int(Kp * error + Ki * integral)
        previous_error = error
        previous_time = current_time


        if abs(error) <= angle_threshold:  # if error is within desired threshold, hover
            print(
                f"Within Threshold. Moving towards detected flow at {logger.flowAngle} degrees with a magnitude of {logger.flow_mag}")
            mc.stop()  # Start Non-Blocking Turn
            time.sleep(0.05)
            #raise InterruptedError("Yaw aligned, landing")
            return True
            # flowMap.append((logger.latest_bx, logger.latest_by, logger.flow_mag, logger.flowAngle))

        #elif (270 <= logger.flowAngle <= 360) or ((desired_flow_angle + angle_threshold) <= logger.flowAngle <= 270):
        elif 180 < logger.flowAngle <= ((desired_flow_angle+360) - angle_threshold):
            right_command = abs(turn_command) / 100  # Convert to angular rate in degrees/s, scale down
            right_command = max(min(right_command, 1.0), 0)  # Scale to Crazyflie range
            print(f"Turning right error {abs(error):2f} Angle: {logger.flowAngle:2f}")
            mc.start_turn_right(right_command * 90)  # Convert to appropriate angular rate
            time.sleep(0.01)
            flowMap.append((logger.latest_bx, logger.latest_by, logger.flow_mag, logger.flowAngle))

        elif ((desired_flow_angle + angle_threshold))<= logger.flowAngle <= 180:
            left_command = abs(turn_command) / 100  # Convert to angular rate in degrees/s, scale down
            left_command = max(min(left_command, 1.0), 0)  # Scale to Crazyflie range
            left_command = abs(left_command)  # Make positive for turn_left function
            print(f"Turning left at {abs(error):2f} Angle: {logger.flowAngle:2f}")
            mc.start_turn_left(left_command * 90)  # Convert to appropriate angular rate
            time.sleep(0.01)
            flowMap.append((logger.latest_bx, logger.latest_by, logger.flow_mag, logger.flowAngle))

        last_error = error
