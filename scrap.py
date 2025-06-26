"""
The following is a restructuring of the ACS module code files from scratch for the sake of organization
(and eventually to improve the efficiency of the code).

Author(s): Alex Kult, Dominik Bartsch (@dominob101)
Date: 6-24-2025
Copyright Alpha Kappa Sigma

"""

class Apogee():
    def flight(
            self,
            filename=r"Raw Flight Data/Full Scale Flight 3.csv",
            tAp=1550,
    ):
        """
        This function is intended to perform continuous apogee prediction for the duration of the flight.

        Args:
            self (object): The object of the class for other functions to use (even though this is the root function.
            tAp (float): The target apogee [meters].
            filename (str): The name of the file to save the flight log to.

        Returns:
            Plots and file?

        """

        ### IMPORTS
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.spatial.transform import Rotation
        import time as simtime

        ### INITIALIZING VARIABLES
        num_points = len(flight_log) # # Number of points
        filtered_flight_log = np.zeros((num_points, 13)) # Matrix to store filtered flight data

        status = "ground" # starts on the ground
        zenith = 0 # Zenith angle is zero
        state = np.zeros((3, 4)) # state matrix

        alt_lst = [] # empty list to store altitude data
        apg_lst = [] # empty list to store apogee data
        t_apg_lst = [] # empty list to store time of apogee
        sim_time_lst = [] # empty list to store simulation time data

        start_time = simtime.time() # starting time of simulation

        # Kalman Filter Initialization
        sigma_process_accel_xy = 0.5
        sigma_accel_sensor_xy = 0.5


        ### DEFINING CHARACTERISTICS
