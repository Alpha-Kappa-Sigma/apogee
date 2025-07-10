"""
This singular file is intended to supplement all
the different .py files to potentially improve efficiency.

Author: Alex Kult (), Dominik Bartsch (@dominob101)
Date: 6-24-2025
Copyright Alpha Kappa Sigma

Current main: flight.py
Exterior python packages:
constants.py, vehicle.py, apogee.py import NONE
convert.py, environment.py, filter.py, math_lib.py, apogee_lib.py import numpy
filter.py imports filterpy
apogee_lib.py, flight.py import scipy
flight.py imports time

Checked:
constants.py
convert.py
environment.py
filter.py

Not yet checked:
math_lib.py
vehicle.py
apogee_lib.py
apogee.py
flight.py

Functions here to stay:
__init__ (initializing function, runs when you create the object/class instance)
temp (temperature as a function of altitude)
update_kalman_filters (updates kalman filters)
teasley_filter (alters quaternions after taking in the rotational acceleration measurements from the gyroscopic sensor on the integrated IMU)

Functions to eliminate/combine:
F2K (converts temperature from degrees Fahrenheit to Kelvin)
quatern2euler & euler2zenith (COMBINE, converts rotation from quaternions to euler angles and then from euler angles to zenith angle)
quatern_prod (quaternion multiplication)
"""
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation
from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import time as simtime

class Apogee():
    """
    This class is intended to serve as the overarching class for the ACS apogee module, 
    supplementing the collection of pythion files in the original ACS module. Accidentally deleted attempt 1.
    Only gotta restart now. We good. Dw guys.

    Authors: Alex Kult (@Alex-Kult), Dominik Bartsch (@dominob101)

    To run this code, make sure you have the numpy, matplotlib, scipy, and filterpy packages installed.
    """
    # --- Initialization ---
    def __init__(self):
        """
        The following function is run upon the declaration of the object.

        attaches all the variables to the Apogee object 
        """
        # __________ CONSTANTS __________
        # --- Inputs assigned to object ---
        # General constants
        self.g = 9.8067 #Acceleration due to gravity [m/s^2]
        self.gamma = 1.4 #Ratio of specific heats for air
        self.R = 287.05 #Specific gas constant for air [J/(kg·K)]
        # Conversion factors
        self.m2ft = 3.28083989501 #meters to feet
        self.mph2ms = 0.44704 #mph to m/s
        # Launch Conditions
        self.temp_ground = 50 #Ground temperature [F]
        self.wind_speed = 10 #Downrange wind speed [mph] (downrange speed is positive)
        self.wind_direction = 270 #[deg] #*Use weather app
        self.launch_direction = 260 #[deg] #*Use compass on phone
        # Environmental Parameters
        self.rough_len = 0.075 #[m] Roughness Length (~0.075 for harvested cropland)
        self.grad_ht = 300 #[m] Gradient Height for open terrain and neutral stability
        self.meas_ht = 10 #[m] Wind Speed Measurement Height (~10m for most weather stations)
        # Kalman filter vars
            # Kalman Filter Initialization (XY)
        self.sigma_process_accel_xy = 0.5
        self.sigma_accel_sensor_xy = 0.5
            # Kalman Filter Initialization (Z) (most critical for altitude/velocity)
        self.sigma_process_accel_z = 1.0 # increase if velocity/position is still too high or drifts
        self.sigma_accel_sensor_z = 0.5 # increase if estimated Z-accel is too noisy
        self.sigma_altimeter_sensor = 0.5 # decrease if estimated Z-position is too noisy or not tracking altimeter well

        # --- Calculated values assigned to object ---
        # Wind Conditions
        self.wind_ground = self.wind_speed*self.mph2ms
        self.wind_vector = self.wind_ground*np.array([np.cos(np.radians(self.wind_direction)), np.sin(np.radians(self.wind_direction))])
        self.launch_vector = np.array([np.cos(np.radians(self.launch_direction)), np.sin(np.radians(self.launch_direction))])
        self.wind_downrange = np.dot(self.wind_vector, self.launch_vector)
        # Gradient wind speed above planetary boundary layer (100-200 meters above ground for open terrain)
        self.grad_speed = self.wind_downrange*np.log(self.grad_ht/self.rough_len)/np.log(self.meas_ht/self.rough_len)
        self.grad_wind = np.array([0, self.grad_speed, 0])

        # __________ KALMAN FILTER INITIALIZATION __________
        # --- Kalman Filter Setup for XY-axes ---
        H_accel_only = np.array([[0., 0., 1.]])
        R_accel_only = np.array([[self.sigma_accel_sensor_xy**2]])
        self.kf_x = KalmanFilter(dim_x=3, dim_z=1)
        self.kf_y = KalmanFilter(dim_x=3, dim_z=1)
        # Set initial common parameters for XY filters
        for kf in [self.kf_x, self.kf_y]:
            kf.H = H_accel_only
            kf.R = R_accel_only
            # Initial State Estimate: [0, 0, 0] (position, velocity, acceleration)
            kf.x = np.array([[0.], [0.], [0.]])
            # Initial Error Covariance: High uncertainty to allow quick convergence
            kf.P = np.array([[1000., 0., 0.], # Position uncertainty
                            [0., 1000., 0.], # Velocity uncertainty
                            [0., 0., 10.]])  # Acceleration uncertainty
        # --- Kalman Filter Setup for Z-axis (Acceleration + Altimeter) ---
        # State vector: [position, velocity, acceleration] (dim_x = 3)
        # Measurement: [acceleration, position] (dim_z = 2)
        self.kf_z = KalmanFilter(dim_x=3, dim_z=2)
        # Measurement Function Matrix (H) for Z-axis: Measures acceleration (index 2) and position (index 0)
        self.kf_z.H = np.array([[0., 0., 1.], # Maps state[2] (acceleration) to measurement[0]
                        [1., 0., 0.]]) # Maps state[0] (position) to measurement[1]
        # Measurement Noise Covariance (R) for Z-axis: 2x2 matrix for accel and altimeter
        # Assuming no correlation between accelerometer and altimeter noise
        self.kf_z.R = np.array([[self.sigma_accel_sensor_z**2, 0.],
                        [0., self.sigma_altimeter_sensor**2]])
        # Initial State Estimate for Z-axis: [0, 0, 0] (position, velocity, acceleration)
        self.kf_z.x = np.array([[0.], [0.], [0.]])
        # Initial Error Covariance for Z-axis: High uncertainty
        self.kf_z.P = np.array([[1000., 0., 0.],
                        [0., 1000., 0.],
                        [0., 0., 10.]])
        self.last_time = None # To store the previous timestamp for dt calculation

    # __________ FUNCTIONS __________
    def temp(self,alt): #Temperature as a function of altitude
        """
        Gives approximation of temperature based on altitude
        Folded in F2K function.

        Args:
            alt (float): Altitude of the vehicle [m]

        Returns:
            tK (float): Temperature [K]
        """
        return ((self.temp_ground - 0.00356*alt*self.m2ft) - 32) / 1.8 + 273.15
    
    def update_kalman_filters(
            self,
            kf_x, 
            kf_y, 
            kf_z, 
            last_time,
            current_time, 
            x_accel_raw, 
            y_accel_raw, 
            z_accel_raw, 
            altimeter_raw,
            sigma_process_accel_xy, 
            sigma_process_accel_z,
    ):
        """
        This function updates the kalman filters

        Args:
            kf_x (class 'filterpy.kalman.KalmanFilter'): Kalman filter for the x-axis (acceleration only) 
            kf_y (class 'filterpy.kalman.KalmanFilter'): Kalman filter for the y-axis (acceleration only)
            kf_z (class 'filterpy.kalman.KalmanFilter'): Kalman filter for the z-axis (combining acceleration and altimeter measurements)
            last_time (int): last time stamp
            current_time (int): current time stamp
            x_accel_raw (float): raw x acceleration
            y_accel_raw (float): raw y acceleration
            z_accel_raw (float): raw z acceleration
            altimeter_raw (float): raw altimeter data
            sigma_process_accel_xy (float): signal noise in XY
            sigma_process_accel_z (float): signal noise in Z
        
        Returns:
            kf_x (class 'filterpy.kalman.KalmanFilter'): Kalman filter for the x-axis (new) 
            kf_y (class 'filterpy.kalman.KalmanFilter'): Kalman filter for the y-axis (new) 
            kf_z (class 'filterpy.kalman.KalmanFilter'): Kalman filter for the z-axis (new) 
            current_time (int): current time stamp
            estimated_states (list): estimated states from the kalman filter at the current time stamp

        """
        if last_time is None:
            current_dt = 0.03 # Arbitrary small initial dt
        else:
            current_dt = current_time - last_time

        # Update F and Q matrices for the current dt for XY axes
        current_F_xy = np.array([[1., current_dt, 0.5 * current_dt**2],
                                [0., 1., current_dt],
                                [0., 0., 1.]])

        current_Q_xy = sigma_process_accel_xy**2 * np.array([[0.25 * current_dt**4, 0.5 * current_dt**3, 0.5 * current_dt**2],
                                                            [0.5 * current_dt**3, current_dt**2, current_dt],
                                                            [0.5 * current_dt**2, current_dt, 1.]])

        # Update F and Q matrices for the current dt for Z-axis
        current_F_z = np.array([[1., current_dt, 0.5 * current_dt**2],
                                [0., 1., current_dt],
                                [0., 0., 1.]])

        current_Q_z = sigma_process_accel_z**2 * np.array([[0.25 * current_dt**4, 0.5 * current_dt**3, 0.5 * current_dt**2],
                                                        [0.5 * current_dt**3, current_dt**2, current_dt],
                                                        [0.5 * current_dt**2, current_dt, 1.]])

        # Assign updated F and Q to each filter
        kf_x.F = current_F_xy
        kf_x.Q = current_Q_xy
        kf_y.F = current_F_xy
        kf_y.Q = current_Q_xy
        kf_z.F = current_F_z
        kf_z.Q = current_Q_z

        # --- Process X-axis ---
        kf_x.predict()
        kf_x.update(np.array([[x_accel_raw]]))

        # --- Process Y-axis ---
        kf_y.predict()
        kf_y.update(np.array([[y_accel_raw]]))

        # --- Process Z-axis (Acceleration + Altimeter) ---
        kf_z.predict()
        combined_z_measurement = np.array([[z_accel_raw],
                                            [altimeter_raw]])
        kf_z.update(combined_z_measurement)

        # Return current estimated states
        #[time, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc]
        estimated_states = [current_time, kf_x.x[0, 0], kf_y.x[0, 0], kf_z.x[0, 0], kf_x.x[1, 0], kf_y.x[1, 0], kf_z.x[1, 0], kf_x.x[2, 0], kf_y.x[2, 0], kf_z.x[2, 0]]

        return kf_x, kf_y, kf_z, current_time, estimated_states
    
    def teasley_filter(self,quat, gyro, dt): 
        """
        This uses the gyroscope's acceleration measurements to find quaternions. Implement more advanced fusion algorithm later.

        Args:
            quat (list): quaternion for current rotation
            gyro (list): angular acceleration measurements to correct the quaternion values
            dt (float): time step
        """
        # Gyroscope interpolation (angular velocity -> change in angular positon)
        omega = np.array([0, gyro[0], gyro[1], gyro[2]])
        dq = 0.5 * np.array(self.quatern_prod(quat, omega))
        # Quaternion with corrected gyro
        q_new = np.array(quat) + dq * dt
        # Normalization
        q_norm = q_new / np.linalg.norm(q_new)
        return q_norm

    def F2K(self,fahrenheit):
        """
        converts to kelvin from degrees fahrenheit.

        Args:
            fahrenheit (float): temperature measurement in degrees Fahrenheit.

        Returns:
            kelvin: temperature measurement in Kelvin
            
        """
        return (fahrenheit - 32) / 1.8 + 273.15
    
    def quatern2euler(self,q): #Converts quaternion to euler angles
        """
        This converts rotation values from quaternions to euler angles.

        Args:
            q (list): quaternion vector

        Returns:
            euler (yaw/psi, pitch/theta, roll/phi): euler angle for current rotation of the vehicle
        """
        # Separating the vector components
        w, x, y, z = q
        # Precompute elements of the rotation matrix
        R11 = 2 * w**2 - 1 + 2 * x**2
        R21 = 2 * (x * y - w * z)
        R31 = 2 * (x * z + w * y)
        R32 = 2 * (y * z - w * x)
        R33 = 2 * w**2 - 1 + 2 * z**2
        # Compute Euler angles
        return np.arctan2(R21, R11), -np.arcsin(R31), np.arctan2(R32, R33) # psi, beta, phi (yaw, pitch, roll)
    
    def euler2zenith(self,euler): 
        """
        This converts rotation values from euler angles to the zenith angle.

        Args:
            euler (list): vector of euler angles for current rotation of the vehicle
        
        Returns:
            zenith (float): current zenith angle of the vehicle
        """
        _, theta, phi = euler
        return np.arccos(np.cos(theta)*np.cos(phi))
    
    def quatern_prod(a, b):
        """
        This performs a quaternion multiplication operation.

        Args:
            a (NumPy array): quaternion vector (w first)
            b (NumPy array): quaternion vector (w first)

        Returns:
            q (NumPy array): quaternion vector (w first)
        """
        # Calculates the quaternion product of quaternion a and b. Not commutative.
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b

        q1 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        q2 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        q3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        q4 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.array([q1, q2, q3, q4])