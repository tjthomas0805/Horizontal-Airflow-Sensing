import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.platformservice import PlatformService

URI = 'radio://0/80/2M'


def main():
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf

        # Initialize PlatformService with your Crazyflie instance
        platform = PlatformService(crazyflie=cf)

        print("Connected — sending ARM request...")
        platform.send_arming_request(True)  # Arming
        print("Armed!")

        # Add your flight logic here, e.g., MotionCommander usage, setpoints, logging, etc.
        time.sleep(5)

        print("Sending DISARM request…")
        platform.send_arming_request(False)  # Disarming
        print("Disarmed.")


if __name__ == '__main__':
    main()
