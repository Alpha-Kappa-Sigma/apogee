"""
The following code is meant to be a demonstration of the working python script for ACS 6dof.

Author: Dominik Bartsch (@dominob101)
Date: 7-11-2025

"""

# imports
from alpha_kappa_sigma import Vehicle

# filename declarations
flight_log_filename = r"Full Scale Flight 1.csv"
cfd_filename = "cfd.csv"

# test run
testRun = Vehicle(cfd_filename=cfd_filename, flight_log_filename=flight_log_filename)
