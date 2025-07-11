The following is the apogee prediction algorithm for the NDRT ACS Design Squad in the 2025-2026 competition/academic year.

Collaborators involved:
Alex Kult (@Alex-Kult) - Initial author;
Dominik Bartsch (@dominob101) - Contributor (Refinement to single file containing class);

If you hope to find the original code developed by Alex Kult, see the branch alex-original
This branch exists as a refinement for that package.
This branch also relies on the assumption that the provided libraries have better/more efficient
    versions of the functions defined in the alex-original branch. Don't know if that's actually
    true but it probably it. Why reinvent the wheel when it STILL WORKS.
In development is the cpp branch, which is just this exact package in C++! (more efficient but a
    lower-level language so more work for us to write it in the first place.)


Things to work on:
Rewrite all the files into one functional class that can be called (IN PROGRESS)
Refine functions in Apogee class (IN PROGRESS)
Figure out what variables need to be referenced outside of __init__ (NOT STARTED)
Rewrite in C++ (NOT STARTED)

Functions here to stay (Apogee Class):
__init__ (initializing function, runs when you create the object/class instance)
temp (temperature as a function of altitude)
update_kalman_filters (updates kalman filters)
teasley_filter (alters quaternions after taking in the rotational acceleration measurements from the gyroscopic sensor on the integrated IMU)

Functions to eliminate/combine (Apogee Class):
F2K (converts temperature from degrees Fahrenheit to Kelvin)
quatern2euler & euler2zenith (COMBINE, converts rotation from quaternions to euler angles and then from euler angles to zenith angle)
quatern_prod (quaternion multiplication)


USAGE:
The main branch relies on the Vehicle python class in alpha_kappa_sigma.py
An example script demonstrating the code is in test.py
