"""
This singular file is intended to supplement all
the different .py files to potentially improve efficiency.

Author(s): Alex Kult (@Alex-Kult), Dominik Bartsch (@dominob101)
Date: 6-24-2025
Copyright Alpha Kappa Sigma

Current main: flight.py
Exterior python packages:
constants.py, vehicle.py, apogee.py import NONE
convert.py, environment.py, filter.py, math_lib.py, apogee_lib.py import numpy
filter.py imports filterpy
apogee_lib.py, flight.py import scipy
flight.py imports time

The following python files have been incorporated:
constants.py
convert.py
environment.py
filter.py
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
acceleration
quatern2zenith (quaternion values to zenith angle for the vehicle)
apogee
sys_drvs

Functions to eliminate/modify:
F2K (converts temperature from degrees Fahrenheit to Kelvin)
quatern_prod (quaternion multiplication)
rk4_step (fourth-order runge-kutta)
"""

from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import numpy as np
import time as simtime


class Vehicle:
    """
    This class is intended to serve as the overarching class for the ACS apogee module,
    supplementing the collection of pythion files in the original ACS module. Accidentally deleted attempt 1.
    Only gotta restart now. We good. Dw guys.

    Authors: Alex Kult (@Alex-Kult), Dominik Bartsch (@dominob101)

    To run this code, make sure you have the numpy, matplotlib, scipy, and filterpy packages installed.
    """

    # --- Initialization ---
    def __init__(
        self,
        cfd_filename: str = None,
        flight_log_filename: str = None,
        in_flight: bool = False,
    ):
        """
        The following method is run upon the declaration of the object.
        It declares a lot of variables, including vehicle parameters,
            environment parameters, wind conditions, Kalman filter
            inputs, and general scientific constants.
        It also initializes the Kalman filters for x, y, & z.
        It also runs through the CFD data
        It also analyzes flight data with plotting if not in live flight.

        Args:
            cfd_filename (str): name of the cfd file that is being read.
            flight_log_filename (str): name of the flight log filename that is being read.
            in_flight (bool): True if this is the microcomputer using it for actual flight.
        """
        # __________ CONSTANTS __________
        # --- Inputs assigned to object ---
        # General constants
        self.g = 9.8067  # Acceleration due to gravity [m/s^2]
        self.gamma = 1.4  # Ratio of specific heats for air
        self.R = 287.05  # Specific gas constant for air [J/(kg·K)]
        self.t_step = 0.1
        # Conversion factors
        self.m2ft = 3.28083989501  # meters to feet
        self.mph2ms = 0.44704  # mph to m/s
        # Launch Conditions
        self.temp_ground = 50  # Ground temperature [F]
        self.wind_speed = 10  # Downrange wind speed [mph] (downrange speed is positive)
        self.wind_direction = 270  # [deg] #*Use weather app
        self.launch_direction = 260  # [deg] #*Use compass on phone
        # Environmental Parameters
        self.rough_len = 0.075  # [m] Roughness Length (~0.075 for harvested cropland)
        self.grad_ht = 300  # [m] Gradient Height for open terrain and neutral stability
        self.meas_ht = (
            10  # [m] Wind Speed Measurement Height (~10m for most weather stations)
        )
        # Vehicle Parameters
        self.cp_cg = (
            0.42  # Distance between center of gravity and center of pressure [m]
        )
        self.dry_mass = 17.625  # Mass of rocket after motor burn [kg]
        self.mom_inertia = (
            8.04  # Moment of inertial of rocket after motor burn [kg*m^2]
        )
        # Kalman filter vars
        # Kalman Filter Initialization (XY)
        self.sigma_process_accel_xy = 0.5
        self.sigma_accel_sensor_xy = 0.5
        # Kalman Filter Initialization (Z) (most critical for altitude/velocity)
        self.sigma_process_accel_z = (
            1.0  # increase if velocity/position is still too high or drifts
        )
        self.sigma_accel_sensor_z = 0.5  # increase if estimated Z-accel is too noisy
        self.sigma_altimeter_sensor = 0.5  # decrease if estimated Z-position is too noisy or not tracking altimeter well

        # --- Calculated values assigned to object ---
        # Wind Conditions
        self.wind_ground = self.wind_speed * self.mph2ms
        self.wind_vector = self.wind_ground * np.array(
            [
                np.cos(np.radians(self.wind_direction)),
                np.sin(np.radians(self.wind_direction)),
            ]
        )
        self.launch_vector = np.array(
            [
                np.cos(np.radians(self.launch_direction)),
                np.sin(np.radians(self.launch_direction)),
            ]
        )
        self.wind_downrange = np.dot(self.wind_vector, self.launch_vector)
        # Gradient wind speed above planetary boundary layer (100-200 meters above ground for open terrain)
        self.grad_speed = (
            self.wind_downrange
            * np.log(self.grad_ht / self.rough_len)
            / np.log(self.meas_ht / self.rough_len)
        )
        self.grad_wind = np.array([0, self.grad_speed, 0])

        # __________ KALMAN FILTER INITIALIZATION __________
        # --- Kalman Filter Setup for XY-axes ---
        H_accel_only = np.array([[0.0, 0.0, 1.0]])
        R_accel_only = np.array([[self.sigma_accel_sensor_xy**2]])
        self.kf_x = KalmanFilter(dim_x=3, dim_z=1)
        self.kf_y = KalmanFilter(dim_x=3, dim_z=1)
        # Set initial common parameters for XY filters
        for kf in [self.kf_x, self.kf_y]:
            kf.H = H_accel_only
            kf.R = R_accel_only
            # Initial State Estimate: [0, 0, 0] (position, velocity, acceleration)
            kf.x = np.array([[0.0], [0.0], [0.0]])
            # Initial Error Covariance: High uncertainty to allow quick convergence
            kf.P = np.array(
                [
                    [1000.0, 0.0, 0.0],  # Position uncertainty
                    [0.0, 1000.0, 0.0],  # Velocity uncertainty
                    [0.0, 0.0, 10.0],
                ]
            )  # Acceleration uncertainty
        # --- Kalman Filter Setup for Z-axis (Acceleration + Altimeter) ---
        # State vector: [position, velocity, acceleration] (dim_x = 3)
        # Measurement: [acceleration, position] (dim_z = 2)
        self.kf_z = KalmanFilter(dim_x=3, dim_z=2)
        # Measurement Function Matrix (H) for Z-axis: Measures acceleration (index 2) and position (index 0)
        self.kf_z.H = np.array(
            [
                [0.0, 0.0, 1.0],  # Maps state[2] (acceleration) to measurement[0]
                [1.0, 0.0, 0.0],
            ]
        )  # Maps state[0] (position) to measurement[1]
        # Measurement Noise Covariance (R) for Z-axis: 2x2 matrix for accel and altimeter
        # Assuming no correlation between accelerometer and altimeter noise
        self.kf_z.R = np.array(
            [[self.sigma_accel_sensor_z**2, 0.0], [0.0, self.sigma_altimeter_sensor**2]]
        )
        # Initial State Estimate for Z-axis: [0, 0, 0] (position, velocity, acceleration)
        self.kf_z.x = np.array([[0.0], [0.0], [0.0]])
        # Initial Error Covariance for Z-axis: High uncertainty
        self.kf_z.P = np.array(
            [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 10.0]]
        )
        self.last_time = None  # To store the previous timestamp for dt calculation

        # __________ CFD DATA INITIALIZATION __________
        # Input data
        if not cfd_filename:
            raise Exception("Please specify the name of the file for the CFD data.")
        self.cfd_data = np.loadtxt(cfd_filename, delimiter=",", dtype=float, skiprows=1)
        # Separates the spreadsheet by column
        acs_angles = np.unique(self.cfd_data[:, 0])
        atk_angles = np.unique(self.cfd_data[:, 1])
        mach_numbers = np.unique(self.cfd_data[:, 2])
        axial_forces = np.full(
            (len(acs_angles), len(atk_angles), len(mach_numbers)), np.nan
        )
        normal_forces = np.full(
            (len(acs_angles), len(atk_angles), len(mach_numbers)), np.nan
        )
        # for each i in the loop, this does...
        for i in range(len(self.cfd_data)):
            # uh...yeah uh...it uhhhhh...
            acs_ang, atk_ang, mach = self.cfd_data[i, :3]
            axial_force = self.cfd_data[i, 3]
            normal_force = self.cfd_data[i, 4]
            # um... *gulp* I don't know what this does
            i_idx = np.where(acs_angles == acs_ang)[0][0]
            j_idx = np.where(atk_angles == atk_ang)[0][0]
            k_idx = np.where(mach_numbers == mach)[0][0]
            # axial and normal forces?
            axial_forces[i_idx, j_idx, k_idx] = axial_force
            normal_forces[i_idx, j_idx, k_idx] = normal_force
        # Ohhhhhhhh so these make interpolations for the axial and normal forces on the vehicle!
        self.axial_interp = RegularGridInterpolator(
            (acs_angles, atk_angles, mach_numbers),
            axial_forces,
            bounds_error=False,
            fill_value=None,
        )
        self.normal_interp = RegularGridInterpolator(
            (acs_angles, atk_angles, mach_numbers),
            normal_forces,
            bounds_error=False,
            fill_value=None,
        )
        # *Audibly inhales* so pwofessow...

        # __________ Flight Data __________
        # checks that there is a name - CHANGE FOR LIVE DATA
        self.apg_target = 1550  # [m]
        if not flight_log_filename:
            if in_flight is True:
                # HERE IS WHERE THE FUTURE PACKAGE FOR THE PI ITSELF GOES
                pass
            else:
                raise Exception(
                    "Please specify the name of the file for the flight log."
                )

        # extracting data from flight log for live processing
        else:
            flight_log = np.loadtxt(
                flight_log_filename, delimiter=",", dtype=float, skiprows=1
            )
            self.filtered_flight_log = np.zeros(
                (len(flight_log), 13)
            )  # [time, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, x_acc_filt, y_acc_filt, z_acc_filt, x_acc_rot, y_acc_rot, z_acc_rot]
            # Starting assumptions
            status = "ground"
            zenith = 0
            state = np.zeros((3, 4))
            # Initialization of vectors (lists) used for plotting
            alt_lst = []
            apg_lst = []
            t_apg_lst = []
            sim_time_lst = []
            # Zeroing time
            start_time = simtime.time()
            # Loop through points in flight log
            for idx in range(len(flight_log)):
                flight_data = flight_log[idx, :]
                # Sensor inputs
                alt_meas = flight_data[1] / self.m2ft
                acc_bno_meas = flight_data[4:7]
                acc_icm_meas = flight_data[17:20]
                quaternion = flight_data[13:17]
                gyro = flight_data[10:13]
                time = flight_data[0]
                # Determine acceleration sensor reading
                if status == "ground" or status == "burn":
                    acc_meas = acc_icm_meas
                else:
                    acc_meas = acc_bno_meas
                    # Consistent sensor frame
                    x, y, z = acc_meas[0], acc_meas[1], acc_meas[2]
                    acc_meas[0] = -y
                    acc_meas[1] = x
                    acc_meas[2] = z
                # Prepare sensor fusion
                if status == "ground":
                    quaternion_old = flight_log[idx - 1, 13:17]  # last quaternion
                if status == "burn" or status == "coast":
                    # At high accelerations, the BNO085's orientation determination is unreliable.
                    # For this reason, manual sensor fusion is used.
                    # NOTE: Gyro drift starts to kick in pretty heavily after around 15-20 seconds.
                    # Calculate quaternions manually using gyro (fusion)
                    # Use more accurate fusion algorithm if we have enough processing power
                    dt = time - self.last_time
                    quaternion = self.teasley_filter(quaternion_old, gyro, dt)
                    quaternion_old = quaternion
                # Sensor frame to body frame
                x, y, z = acc_meas[0], acc_meas[1], acc_meas[2]
                acc_meas[0] = z
                acc_meas[1] = y
                acc_meas[2] = x
                # Calculate euler angles and zenith angle [rad]
                zenith_old = zenith
                zenith = self.quatern2zenith(quaternion)
                # Body frame to global frame
                r = Rotation.from_euler("y", np.degrees(zenith) - 90, degrees=True)
                acc_i = r.apply(acc_meas)
                acc_i[2] -= self.g
                ax_meas, ay_meas, az_meas = acc_i
                # Use kalman filter on acceleration and altitude data
                self.kf_x, self.kf_y, self.kf_z, self.last_time, current_estimates = (
                    self.update_kalman_filters(
                        self.kf_x,
                        self.kf_y,
                        self.kf_z,
                        self.last_time,
                        time,
                        ax_meas,
                        ay_meas,
                        az_meas,
                        alt_meas,
                        self.sigma_process_accel_xy,
                        self.sigma_process_accel_z,
                    )
                )
                # Store filtered results
                self.filtered_flight_log[idx, :] = np.hstack((current_estimates, acc_i))
                pos_z = current_estimates[3]
                vel_z = current_estimates[6]
                acc_z = current_estimates[9]
                # Apogee Prediction
                if status == "coast":
                    loop_start_time = simtime.time()
                    # Downrange Conditions
                    pos_horz = np.linalg.norm(np.array(current_estimates[1:3]))
                    vel_horz = np.linalg.norm(np.array(current_estimates[4:6]))
                    # Estimate angular velocty along pitch axis (improve later with gyro sensor readings)
                    omega = (zenith - zenith_old) / dt
                    # Initializing state matrix
                    state[:2, 0] = [pos_z, pos_horz]
                    state[:2, 1] = [vel_z, vel_horz]
                    state[2, 2] = zenith
                    state[2, 3] = omega
                    # Predict apogee
                    apogee = self.apogee_pred(state)
                    loop_end_time = simtime.time()
                    loop_time = loop_end_time - loop_start_time
                    # Append info to lists for plotting
                    alt_lst.append(pos_z)
                    apg_lst.append(apogee)
                    t_apg_lst.append(time)
                    sim_time_lst.append(loop_time)
                # State determination
                if acc_z > 5 and abs(pos_z) > 1 and status == "ground":
                    status = "burn"
                    t_burn = time
                    print(f"Engine burn at t = {time:.4f} seconds.")
                elif (
                    acc_z < 0
                    and pos_z < self.apg_target
                    and vel_z > 0
                    and status == "burn"
                ):
                    status = "coast"
                    t_burnout = time
                    print(f"Engine burnout at t = {time:.4f} seconds.")
                elif acc_z < 0 and pos_z >= self.apg_target and status == "coast":
                    status = "overshoot"
                    print(f"Overshoot at t = {time:.4f} seconds.")
                elif (
                    acc_z < 0
                    and vel_z <= 0
                    and (status == "overshoot" or status == "coast")
                ):
                    status = "descent"
                    apogee = pos_z
                    t_apogee = time
                    print(
                        f"Apogee of {apogee:.4f} m reached at t = {t_apogee:.4f} seconds."
                    )

            # Calculating Simulation Time and Frequency Information
            end_time = simtime.time()
            tot_time = end_time - start_time
            print(f"Total Simulation Time: {tot_time:.4f} seconds.")

            t_sim_lst = [
                t_apg_lst[t] for t in range(len(sim_time_lst)) if sim_time_lst[t] != 0
            ]
            sim_time_lst = [val for val in sim_time_lst if val != 0]
            hertz_lst = [1 / t if t != 0 else 0 for t in sim_time_lst]
            print(f"Minimum Simulation Hertz: {np.min(hertz_lst):.4f} Hz.")

            # --- Plotting ---
            # Plotting Apogee Prediction Throughout Coast
            plt.plot(t_apg_lst, alt_lst, label="Altitude")
            plt.plot(t_apg_lst, apg_lst, label="Predicted Apogee")
            plt.axhline(apogee, label="Apogee", color="g")
            plt.xlabel("Time [s]")
            plt.ylabel("Altitude [m]")
            plt.title("Apogee Prediction")
            plt.legend()
            plt.grid()
            plt.show()

            # Plotting Filtered Position, Velocity, and Acceleration Data (Global Frame)
            plt.figure(figsize=(15, 15))
            times = self.filtered_flight_log[:, 0]

            # X-axis plots
            plt.subplot(3, 3, 1)
            plt.plot(
                times, self.filtered_flight_log[:, 1], label="Estimated X Position"
            )
            plt.axvline(t_burn, label="Burn", color="k")
            plt.axvline(t_burnout, label="Burnout", color="r")
            plt.axvline(t_apogee, label="Apogee", color="g")
            plt.ylabel("X Position [m]")
            plt.title("X-axis Estimates")
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 4)
            plt.plot(
                times, self.filtered_flight_log[:, 4], label="Estimated X Velocity"
            )
            plt.axvline(t_burn, label="Burn", color="k")
            plt.axvline(t_burnout, label="Burnout", color="r")
            plt.axvline(t_apogee, label="Apogee", color="g")
            plt.ylabel("X Velocity [m/s]")
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 7)
            plt.plot(
                times,
                self.filtered_flight_log[:, 10],
                label="Raw X Acceleration",
                alpha=0.6,
            )
            plt.plot(
                times, self.filtered_flight_log[:, 7], label="Estimated X Acceleration"
            )
            plt.axvline(t_burn, label="Burn", color="k")
            plt.axvline(t_burnout, label="Burnout", color="r")
            plt.axvline(t_apogee, label="Apogee", color="g")
            plt.xlabel("Time [s]")
            plt.ylabel("X Acceleration [m/s^2]")
            plt.legend()
            plt.grid()

            # Y-axis plots
            plt.subplot(3, 3, 2)
            plt.plot(
                times, self.filtered_flight_log[:, 2], label="Estimated Y Position"
            )
            plt.axvline(t_burn, label="Burn", color="k")
            plt.axvline(t_burnout, label="Burnout", color="r")
            plt.axvline(t_apogee, label="Apogee", color="g")
            plt.ylabel("Y Position [m]")
            plt.title("Y-axis Estimates")
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 5)
            plt.plot(
                times, self.filtered_flight_log[:, 5], label="Estimated Y Velocity"
            )
            plt.axvline(t_burn, label="Burn", color="k")
            plt.axvline(t_burnout, label="Burnout", color="r")
            plt.axvline(t_apogee, label="Apogee", color="g")
            plt.ylabel("Y Velocity [m/s]")
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 8)
            plt.plot(
                times,
                self.filtered_flight_log[:, 11],
                label="Raw Y Acceleration",
                alpha=0.6,
            )
            plt.plot(
                times, self.filtered_flight_log[:, 8], label="Estimated Y Acceleration"
            )
            plt.axvline(t_burn, label="Burn", color="k")
            plt.axvline(t_burnout, label="Burnout", color="r")
            plt.axvline(t_apogee, label="Apogee", color="g")
            plt.xlabel("Time [s]")
            plt.ylabel("Y Acceleration [m/s^2]")
            plt.legend()
            plt.grid()

            # Z-axis plots
            plt.subplot(3, 3, 3)
            plt.plot(
                times, flight_log[:, 1] / self.m2ft, label="Raw Altimeter", alpha=0.6
            )
            plt.plot(
                times, self.filtered_flight_log[:, 3], label="Estimated Z Position"
            )
            plt.axvline(t_burn, label="Burn", color="k")
            plt.axvline(t_burnout, label="Burnout", color="r")
            plt.axvline(t_apogee, label="Apogee", color="g")
            plt.axhline(apogee, label="Apogee", color="g")
            plt.axhline(self.apg_target, label="Apogee Target", color="y")
            plt.ylabel("Z Position [m]")
            plt.title("Z-axis Estimates")
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 6)
            plt.plot(
                times, self.filtered_flight_log[:, 6], label="Estimated Z Velocity"
            )
            plt.axvline(t_burn, label="Burn", color="k")
            plt.axvline(t_burnout, label="Burnout", color="r")
            plt.axvline(t_apogee, label="Apogee", color="g")
            plt.ylabel("Z Velocity [m/s]")
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 9)
            plt.plot(
                times,
                self.filtered_flight_log[:, 12],
                label="Raw Z Acceleration",
                alpha=0.6,
            )
            plt.plot(
                times, self.filtered_flight_log[:, 9], label="Estimated Z Acceleration"
            )
            plt.axvline(t_burn, label="Burn", color="k")
            plt.axvline(t_burnout, label="Burnout", color="r")
            plt.axvline(t_apogee, label="Apogee", color="g")
            plt.xlabel("Time [s]")
            plt.ylabel("Z Acceleration [m/s^2]")
            plt.legend()
            plt.grid()

            plt.tight_layout()
            plt.show()

            # Apogee Prediction Frequency Plot
            plt.plot(t_sim_lst, hertz_lst)
            plt.xlabel("Time [s]")
            plt.ylabel("Hertz [s^(-1)]")
            plt.title("Apogee Prediction Simulation Frequency")
            plt.grid()
            plt.show()

    # __________ FUNCTIONS __________
    def temp(self, alt):  # Temperature as a function of altitude
        """
        Gives approximation of temperature based on altitude
        Folded in F2K function.

        Args:
            alt (float): Altitude of the vehicle [m]

        Returns:
            tK (float): Temperature [K]
        """
        return ((self.temp_ground - 0.00356 * alt * self.m2ft) - 32) / 1.8 + 273.15

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
            current_dt = 0.03  # Arbitrary small initial dt
        else:
            current_dt = current_time - last_time

        # Update F and Q matrices for the current dt for XY axes
        current_F_xy = np.array(
            [
                [1.0, current_dt, 0.5 * current_dt**2],
                [0.0, 1.0, current_dt],
                [0.0, 0.0, 1.0],
            ]
        )

        current_Q_xy = sigma_process_accel_xy**2 * np.array(
            [
                [0.25 * current_dt**4, 0.5 * current_dt**3, 0.5 * current_dt**2],
                [0.5 * current_dt**3, current_dt**2, current_dt],
                [0.5 * current_dt**2, current_dt, 1.0],
            ]
        )

        # Update F and Q matrices for the current dt for Z-axis
        current_F_z = np.array(
            [
                [1.0, current_dt, 0.5 * current_dt**2],
                [0.0, 1.0, current_dt],
                [0.0, 0.0, 1.0],
            ]
        )

        current_Q_z = sigma_process_accel_z**2 * np.array(
            [
                [0.25 * current_dt**4, 0.5 * current_dt**3, 0.5 * current_dt**2],
                [0.5 * current_dt**3, current_dt**2, current_dt],
                [0.5 * current_dt**2, current_dt, 1.0],
            ]
        )

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
        combined_z_measurement = np.array([[z_accel_raw], [altimeter_raw]])
        kf_z.update(combined_z_measurement)

        # Return current estimated states
        # [time, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc]
        estimated_states = [
            current_time,
            kf_x.x[0, 0],
            kf_y.x[0, 0],
            kf_z.x[0, 0],
            kf_x.x[1, 0],
            kf_y.x[1, 0],
            kf_z.x[1, 0],
            kf_x.x[2, 0],
            kf_y.x[2, 0],
            kf_z.x[2, 0],
        ]

        return kf_x, kf_y, kf_z, current_time, estimated_states

    def teasley_filter(self, quat, gyro, dt):
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

    def acceleration(self, acs_ang, state):
        """
        This function acceleration and angular acceleration based on the angles?

        Args:
            acs_ang ()
            state ()

        Returns:
            acc ()
            ang_acc ()

        """
        # Disecting state matrix
        alt = state[0, 0]
        vel = state[:, 1]
        zenith = state[2, 2]

        # Gravitational Force
        grav_acc = np.array([-self.g, 0, 0])

        # Aerodynamic State
        temp_k = self.temp(alt)
        speed_of_sound = np.sqrt(self.gamma * self.R * temp_k)

        vel_rel = vel - self.grad_wind
        mach = np.linalg.norm(vel_rel) / speed_of_sound

        # Calculate Aerodynamic Forces
        if mach >= 0.025:
            # Angle of Attack using zenith angle and velocity angle
            lift_state = True
            atk_ang = zenith - abs(np.arctan(vel_rel[1] / vel_rel[0]))
            if atk_ang < 0:
                lift_state = False
            atk_ang = abs(atk_ang)

            # Aerodynamic Forces and Moments
            aero_point = np.array([acs_ang, np.degrees(atk_ang), mach])
            axial_force_mag = self.axial_interp(aero_point)[0]
            normal_force_mag = self.normal_interp(aero_point)[0]

            axial_force = axial_force_mag * np.array(
                [-np.cos(zenith), -np.sin(zenith), 0]
            )
            normal_force = normal_force_mag * np.array(
                [-np.sin(zenith), np.cos(zenith), 0]
            )

            aero_mom = -normal_force_mag * self.cp_cg
            aero_mom *= 0.2  # Dampening Torque (From Teasley)

            ang_acc = aero_mom / self.mom_inertia

            # Opposite Direction of lift force and moment if angle of attack is negative
            if not lift_state:
                normal_force = -normal_force
                ang_acc = -ang_acc

            aero_acc = (axial_force + normal_force) / self.dry_mass

            acc = grav_acc + aero_acc
        else:
            acc = grav_acc
            ang_acc = 0

        return acc, ang_acc

    def apogee_pred(self, state):
        """
        apogee prediction based on current state

        Args:
            state ()

        Returns:
            apogee ()
        """
        alt_lst = [state[0, 0]]
        time = 0

        while state[0, 1] > 0:
            state = self.rk4_step(state, self.t_step)
            time += self.t_step
            alt_lst.append(state[0, 0])

        apogee = alt_lst[-1]

        return apogee

    def sys_drvs(self, state):
        """
        This function...

        Args:
            state ()

        Returns:
            state_drv ()

        """
        vel = state[:, 1]
        ang_vel = state[2, 3]
        acc, ang_acc = self.acceleration(0, state)

        dx_dt = vel
        dv_dt = acc
        dang_dt = ang_vel
        dangvel_dt = ang_acc

        state_drv = np.zeros((3, 4))
        state_drv[:, 0] = dx_dt
        state_drv[:, 1] = dv_dt
        state_drv[2, 2] = dang_dt
        state_drv[2, 3] = dangvel_dt
        return state_drv

    def F2K(self, fahrenheit):
        """
        converts to kelvin from degrees fahrenheit.

        Args:
            fahrenheit (float): temperature measurement in degrees Fahrenheit.

        Returns:
            kelvin: temperature measurement in Kelvin

        """
        return (fahrenheit - 32) / 1.8 + 273.15

    def quatern2zenith(self, q):  # Converts quaternion to euler angles
        """
        This converts rotation values from quaternions to the zenith angle.

        Args:
            q (list): quaternion vector

        Returns:
            zenith (float): current zenith angle of the vehicle
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
        __, beta, phi = (
            np.arctan2(R21, R11),
            -np.arcsin(R31),
            np.arctan2(R32, R33),
        )  # psi, beta, phi (yaw, pitch, roll)
        # zenith angle from euler angles
        return np.arccos(np.cos(beta) * np.cos(phi))

    def quatern_prod(self, a, b):
        """
        This performs a quaternion multiplication operation. NOT COMMUTATIVE

        Args:
            a (NumPy array): quaternion vector (w first)
            b (NumPy array): quaternion vector (w first)

        Returns:
            q (NumPy array): quaternion vector (w first)
        """
        # Separating vectors into components
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        # multiplication for each resultant value in the vector
        q1 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        q2 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        q3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        q4 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([q1, q2, q3, q4])

    def rk4_step(self, state, t_step):
        """
        Fourth-order runge-kutta approximation of state

        Args:
            state ()
            t_step ()

        Returns:
            state_new ()
        """
        # four steps
        k1 = self.sys_drvs(state)
        k2 = self.sys_drvs(state + 0.5 * k1 * t_step)
        k3 = self.sys_drvs(state + 0.5 * k2 * t_step)
        k4 = self.sys_drvs(state + k3 * t_step)
        # final combined calculation
        return state + t_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
