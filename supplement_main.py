"""
This singular file is intended to supplement all
the different .py files to potentially improve efficiency.

Author: Alex Kult (), Dominik Bartsch (@dominob101)
Date: 6-24-2025
Copyright Alpha Kappa Sigma

Current main: flight.py

Is used:
flight.py references apogee.py
flight.py references filter.py
flight.py references convert.py
flight.py references math_lib.py
flight.py references constants.py
apogee.py references apogee_lib.py
apogee_lib.py references constants
apogee_lib.py references vehicle
apogee_lib.py references environment
apogee_lib.py references math_lib
vehicle references NONE
constants references NONE

"""
from scipy.interpolate import RegularGridInterpolator # apogee_lib
from scipy.spatial.transform import Rotation # flight
import numpy as np # apogee_lib, flight, convert
import matplotlib.pyplot as plt # flight
import time as simtime # flight

class Apogee():
    """
    This class is intended to serve as the overarching class for the ACS apogee module, supplementing the collection of pythion files in the original ACS module.

    To run this code, you should only need to run the following modules:
    scipy
    numpy
    matplotlib

    """

    # --- Imports ---

    # --- Initialization ---
    def __init__(self,cfd_csv_file = r"cfd.csv"):


    # --- Functions ---
    def F2K(self,fahrenheit):  # fahrenheit to kelvin
        kelvin = (fahrenheit - 32) / 1.8 + 273.15
        return kelvin

    def acceleration(self,acs_ang, state):
        # Disecting state matrix
        alt = state[0, 0]
        vel = state[:, 1]
        zenith = state[2, 2]

        # Gravitational Force
        grav_acc = np.array([-c.g, 0, 0])

        # Aerodynamic State
        temp_k = e.temp(alt)
        speed_of_sound = np.sqrt(c.gamma * c.R * temp_k)

        vel_rel = vel - e.grad_wind
        mach = mth.mag(vel_rel) / speed_of_sound

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
            axial_force_mag = axial_interp(aero_point)[0]
            normal_force_mag = normal_interp(aero_point)[0]

            axial_force = axial_force_mag * np.array([-np.cos(zenith), -np.sin(zenith), 0])
            normal_force = normal_force_mag * np.array([-np.sin(zenith), np.cos(zenith), 0])

            aero_mom = -normal_force_mag * v.cp_cg
            aero_mom *= 0.2  # Dampening Torque (From Teasley)

            ang_acc = aero_mom / v.mom_inertia

            # Opposite Direction of lift force and moment if angle of attack is negative
            if not lift_state:
                normal_force = -normal_force
                ang_acc = -ang_acc

            aero_acc = (axial_force + normal_force) / v.dry_mass

            acc = grav_acc + aero_acc
        else:
            acc = grav_acc
            ang_acc = 0

        return acc, ang_acc

    # State Derivatives
    def sys_drvs(self,state):
        vel = state[:, 1]
        ang_vel = state[2, 3]
        acc, ang_acc = acceleration(0, state)

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

    # Uses the RK4 method to numerically integrate state over time
    def rk4_step(self, state, t_step):

        k1 = sys_drvs(state)
        k2 = sys_drvs(state + 0.5 * k1 * t_step)
        k3 = sys_drvs(state + 0.5 * k2 * t_step)
        k4 = sys_drvs(state + k3 * t_step)

        state_new = state + t_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return state_new

    def quatern2euler(self, q):  # Converts quaternion to euler angles
        # Separate Components
        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]

        # Precompute elements of the rotation matrix
        R11 = 2 * w ** 2 - 1 + 2 * x ** 2
        R21 = 2 * (x * y - w * z)
        R31 = 2 * (x * z + w * y)
        R32 = 2 * (y * z - w * x)
        R33 = 2 * w ** 2 - 1 + 2 * z ** 2

        # Compute Euler angles
        phi = np.arctan2(R32, R33)
        theta = -np.arctan(R31 / np.sqrt(1 - R31 ** 2))
        psi = np.arctan2(R21, R11)

        return psi, theta, phi  # Yaw, pitch, roll

    def euler2zenith(self, euler):  # Converts euler angle to zenith angle
        theta = euler[1]
        phi = euler[2]
        zenith = np.arccos(np.cos(theta) * np.cos(phi))
        return zenith

    def apogee_pred(state,t_step = 0.1):
        """
        Predict apogee using Runge-Kutta 4th order method.

        :return:
        """
        alt_lst = [state[0, 0]]
        time = 0

        while state[0, 1] > 0:
            state = rk4_step(state, t_step)
            time += t_step
            alt_lst.append(state[0, 0])

        apogee = alt_lst[-1]

        return apogee


    def flight(self, apg_target,cfd_csv_file = r"cfd.csv",flight_log_file,pred=False,pva=False):
        """
        During actual flight, run this file to perform apogee prediction and state determination (this is the main function).

        Args:
            apg_target (float): Target apogee [m].
            flight_log_file (str): Path to flight log file.
            cfd_csv_file = r"cfd.csv"
            pred (bool): Whether to plot apogee prediction throughout coast.
            pva (bool): Whether to plot position, velocity, and acceleration data.

        Returns:
            n/a

        """
        # --- Imports ---
        from scipy.interpolate import RegularGridInterpolator  # apogee_lib
        from scipy.spatial.transform import Rotation  # flight
        import numpy as np  # apogee_lib, flight, convert
        import matplotlib.pyplot as plt  # flight
        import time as simtime  # flight

        # --- Initialization --- (From
        # CFD Force Interpolation Initialization
        cfd_data = np.loadtxt(cfd_csv_file, delimiter=",", dtype=float, skiprows=1)

        acs_angles = np.unique(cfd_data[:, 0])
        atk_angles = np.unique(cfd_data[:, 1])
        mach_numbers = np.unique(cfd_data[:, 2])

        axial_forces = np.full((len(acs_angles), len(atk_angles), len(mach_numbers)), np.nan)
        normal_forces = np.full((len(acs_angles), len(atk_angles), len(mach_numbers)), np.nan)

        for i in range(len(cfd_data)):
            acs_ang, atk_ang, mach = cfd_data[i, :3]
            axial_force = cfd_data[i, 3]
            normal_force = cfd_data[i, 4]

            i_idx = np.where(acs_angles == acs_ang)[0][0]
            j_idx = np.where(atk_angles == atk_ang)[0][0]
            k_idx = np.where(mach_numbers == mach)[0][0]

            axial_forces[i_idx, j_idx, k_idx] = axial_force
            normal_forces[i_idx, j_idx, k_idx] = normal_force

        axial_interp = RegularGridInterpolator((acs_angles, atk_angles, mach_numbers), axial_forces, bounds_error=False,
                                               fill_value=None)
        normal_interp = RegularGridInterpolator((acs_angles, atk_angles, mach_numbers), normal_forces,
                                                bounds_error=False,
                                                fill_value=None)

        # --- Files ---
        flight_log = np.loadtxt(flight_log_file, delimiter=",", dtype=float, skiprows=1)

        # --- Initialization ---
        num_points = len(flight_log)
        filtered_flight_log = np.zeros((num_points,
                                        13))  # [time, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, x_acc_filt, y_acc_filt, z_acc_filt, x_acc_rot, y_acc_rot, z_acc_rot]

        status = "ground"
        zenith = 0
        state = np.zeros((3, 4))

        alt_lst = []
        apg_lst = []
        t_apg_lst = []
        sim_time_lst = []

        start_time = simtime.time()

        # Kalman Filter Initialization
        sigma_process_accel_xy = 0.5
        sigma_accel_sensor_xy = 0.5

        # For Z-axis (most critical for altitude/velocity)
        # If velocity/position is still too high or drifts, increase sigma_process_accel_z.
        # If estimated Z-accel is too noisy, increase sigma_accel_sensor_z.
        # If estimated Z-position is too noisy or not tracking altimeter well, decrease sigma_altimeter_sensor.
        sigma_process_accel_z = 1.0
        sigma_accel_sensor_z = 0.5
        sigma_altimeter_sensor = 0.5

        kf_x, kf_y, kf_z, last_time = kal.initialize_kalman_filters(sigma_accel_sensor_xy,
                                                                    sigma_accel_sensor_z,
                                                                    sigma_altimeter_sensor)

        # Loop through points in flight log
        for idx in range(num_points):
            flight_data = flight_log[idx, :]

            # Sensor inputs
            alt_meas = flight_data[1] / c.m2ft
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
                dt = time - last_time
                quaternion = kal.teasley_filter(quaternion_old, gyro, dt)
                quaternion_old = quaternion

            # Sensor frame to body frame
            x, y, z = acc_meas[0], acc_meas[1], acc_meas[2]
            acc_meas[0] = z
            acc_meas[1] = y
            acc_meas[2] = x

            # Calculate euler angles and zenith angle [rad]
            zenith_old = zenith
            euler = con.quatern2euler(quaternion)
            yaw, pitch, roll = euler
            zenith = con.euler2zenith(euler)

            # Body frame to global frame
            r = Rotation.from_euler("y", np.degrees(zenith) - 90, degrees=True)
            acc_i = r.apply(acc_meas)
            acc_i[2] -= c.g

            ax_meas, ay_meas, az_meas = acc_i

            # Use kalman filter on acceleration and altitude data
            kf_x, kf_y, kf_z, last_time, current_estimates = kal.update_kalman_filters(kf_x, kf_y, kf_z, last_time,
                                                                                       time, ax_meas, ay_meas, az_meas,
                                                                                       alt_meas,
                                                                                       sigma_process_accel_xy,
                                                                                       sigma_process_accel_z)

            # Store filtered results
            filtered_flight_log[idx, :] = np.hstack((current_estimates, acc_i))
            pos_z = current_estimates[3]
            vel_z = current_estimates[6]
            acc_z = current_estimates[9]

            # Apogee Prediction
            if status == "coast":
                loop_start_time = simtime.time()

                # Downrange Conditions
                pos_horz = mth.mag(np.array(current_estimates[1:3]))
                vel_horz = mth.mag(np.array(current_estimates[4:6]))

                # Estimate angular velocty along pitch axis (improve later with gyro sensor readings)
                omega = (zenith - zenith_old) / dt

                # Initializing state matrix
                state[:2, 0] = [pos_z, pos_horz]
                state[:2, 1] = [vel_z, vel_horz]
                state[2, 2] = zenith
                state[2, 3] = omega

                # Predict apogee
                apogee = apg.apogee_pred(state)

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
            elif acc_z < 0 and pos_z < apg_target and vel_z > 0 and status == "burn":
                status = "coast"
                t_burnout = time
                print(f"Engine burnout at t = {time:.4f} seconds.")
            elif acc_z < 0 and pos_z >= apg_target and status == "coast":
                status = "overshoot"
                print(f"Overshoot at t = {time:.4f} seconds.")
            elif acc_z < 0 and vel_z <= 0 and (status == "overshoot" or status == "coast"):
                status = "descent"
                apogee = pos_z
                t_apogee = time
                print(f"Apogee of {apogee:.4f} m reached at t = {t_apogee:.4f} seconds.")

        # Calculating Simulation Time and Frequency Information
        end_time = simtime.time()
        tot_time = end_time - start_time
        print(f"Total Simulation Time: {tot_time:.4f} seconds.")

        t_sim_lst = [t_apg_lst[t] for t in range(len(sim_time_lst)) if sim_time_lst[t] != 0]
        sim_time_lst = [val for val in sim_time_lst if val != 0]
        hertz_lst = [1 / t if t != 0 else 0 for t in sim_time_lst]
        print(f"Minimum Simulation Hertz: {np.min(hertz_lst):.4f} Hz.")

        # --- Plotting ---
        # Plotting Apogee Prediction Throughout Coast
        if pred == True:
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
        if pva == True:
            plt.figure(figsize=(15, 15))
            times = filtered_flight_log[:, 0]

            # X-axis plots
            plt.subplot(3, 3, 1)
            plt.plot(times, filtered_flight_log[:, 1], label="Estimated X Position")
            plt.axvline(t_burn, label="Burn", color="k")
            plt.axvline(t_burnout, label="Burnout", color="r")
            plt.axvline(t_apogee, label="Apogee", color="g")
            plt.ylabel("X Position [m]")
            plt.title("X-axis Estimates")
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 4)
            plt.plot(times, filtered_flight_log[:, 4], label="Estimated X Velocity")
            plt.axvline(t_burn, label="Burn", color="k")
            plt.axvline(t_burnout, label="Burnout", color="r")
            plt.axvline(t_apogee, label="Apogee", color="g")
            plt.ylabel("X Velocity [m/s]")
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 7)
            plt.plot(times, filtered_flight_log[:, 10], label="Raw X Acceleration", alpha=0.6)
            plt.plot(times, filtered_flight_log[:, 7], label="Estimated X Acceleration")
            plt.axvline(t_burn, label="Burn", color="k")
            plt.axvline(t_burnout, label="Burnout", color="r")
            plt.axvline(t_apogee, label="Apogee", color="g")
            plt.xlabel("Time [s]")
            plt.ylabel("X Acceleration [m/s^2]")
            plt.legend()
            plt.grid()

            # Y-axis plots
            plt.subplot(3, 3, 2)
            plt.plot(times, filtered_flight_log[:, 2], label="Estimated Y Position")
            plt.axvline(t_burn, label="Burn", color="k")
            plt.axvline(t_burnout, label="Burnout", color="r")
            plt.axvline(t_apogee, label="Apogee", color="g")
            plt.ylabel("Y Position [m]")
            plt.title("Y-axis Estimates")
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 5)
            plt.plot(times, filtered_flight_log[:, 5], label="Estimated Y Velocity")
            plt.axvline(t_burn, label="Burn", color="k")
            plt.axvline(t_burnout, label="Burnout", color="r")
            plt.axvline(t_apogee, label="Apogee", color="g")
            plt.ylabel("Y Velocity [m/s]")
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 8)
            plt.plot(times, filtered_flight_log[:, 11], label="Raw Y Acceleration", alpha=0.6)
            plt.plot(times, filtered_flight_log[:, 8], label="Estimated Y Acceleration")
            plt.axvline(t_burn, label="Burn", color="k")
            plt.axvline(t_burnout, label="Burnout", color="r")
            plt.axvline(t_apogee, label="Apogee", color="g")
            plt.xlabel("Time [s]")
            plt.ylabel("Y Acceleration [m/s^2]")
            plt.legend()
            plt.grid()

            # Z-axis plots
            plt.subplot(3, 3, 3)
            plt.plot(times, flight_log[:, 1] / c.m2ft, label="Raw Altimeter", alpha=0.6)
            plt.plot(times, filtered_flight_log[:, 3], label="Estimated Z Position")
            plt.axvline(t_burn, label="Burn", color="k")
            plt.axvline(t_burnout, label="Burnout", color="r")
            plt.axvline(t_apogee, label="Apogee", color="g")
            plt.axhline(apogee, label="Apogee", color="g")
            plt.axhline(apg_target, label="Apogee Target", color="y")
            plt.ylabel("Z Position [m]")
            plt.title("Z-axis Estimates")
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 6)
            plt.plot(times, filtered_flight_log[:, 6], label="Estimated Z Velocity")
            plt.axvline(t_burn, label="Burn", color="k")
            plt.axvline(t_burnout, label="Burnout", color="r")
            plt.axvline(t_apogee, label="Apogee", color="g")
            plt.ylabel("Z Velocity [m/s]")
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 9)
            plt.plot(times, filtered_flight_log[:, 12], label="Raw Z Acceleration", alpha=0.6)
            plt.plot(times, filtered_flight_log[:, 9], label="Estimated Z Acceleration")
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