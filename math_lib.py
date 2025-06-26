"""
math_lib.py
An reference library of math functions for all ACS module code files
Date: 6-9-2025


Author: Alex Kult
Copyright Alpha Kappa Sigma
"""

# --- Imports ---
import numpy as np

# --- Functions ---
def mag(vector):
    return np.sqrt(np.sum(vector**2))

def quatern_prod(a, b):
    
    # Calculates the quaternion product of quaternion a and b. Not commutative.
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b

    q1 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    q2 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    q3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    q4 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([q1, q2, q3, q4])