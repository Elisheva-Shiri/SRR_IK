#  <!-- a code that calculate distance, spheare, circuil and Trim points. Using python in a few steps:
# 1. I have a point (x_1,y_1,z_3), I have a length d_1 and Angle phi_1. I want to find the point (x_2,y_2,z_2) that is in a distance d_1 and angle phi_1 clock wise  from (x_1,y_1,z_1). 
# 2. I have (x_2,y_2,z_2) and (x_b,y_b,z_b) I want to calculate d_2 the distance between both of them.
# 3. I have d_2, d_3, d_4 and I want to calculate the andle between  d_3, d_4 using cos-sin smathematicle sentense.

# (x_1,y_1,z_3), (x_b,y_b,z_b), phi_1, d_3, d_4 are known variblse  -->

## #----imports
import math
from math import sqrt, acos, degrees
import numpy as np
# from sympy import symbols, Eq, solve
import csv
import os
from datetime import datetime

# TODO: should i use diffrent val base the real size?
# xL, yL, zL = 2.0, 1.0, 3.0   # Left coordinates of the first point
# xR, yR, zR = 3.0, 5.0, 12.0  # Right coordinates of the first point
# xT, yT, zT = 3.0, 5.0, 12.0  # Top coordinates of the first point

# Define known variables in cm and dgrees
d1 = 8.0  # Base the Mersedes.stl component
xb, yb, zb = 2.0, 1.0, 3.0  
x1, y1, z1 = 3.0, 5.0, 12.0  # Example coordinates of the first point

phi1 = 90.0  # Angle in degrees clockwise
d3 = 8.0  # Example value for distance d3
d4 = 8.0  # Example value for distance d4

def calculate_second_point(x1, y1, z1, d1, phi1 = 0):
    # Convert phi1 to radians for calculations
    phi1_rad = math.radians(phi1)
    x2 = x1 + d1 * math.cos(phi1_rad)
    y2 = y1 + d1 * math.sin(phi1_rad)
    z2 = z1  # Assuming z2 remains the same since no vertical angle is provided
    return x2, y2, z2

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points of any dimension
    point1, point2: tuples/lists containing coordinates of equal length
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions")
        
    sum_squares = 0
    for i in range(len(point1)):
        sum_squares += (point2[i] - point1[i])**2
    return math.sqrt(sum_squares)


def calculate_angle_cosine(d_1, d_2, d_3):
    Err = "Invalid input values for calculating angle. Check distances."
    if d_1 <= 0 or d_2 <= 0 or d_3 <= 0:
        return Err, Err  # invalid sides
    
    # Triangle inequality check
    if d_1 + d_2 <= d_3 or d_1 + d_3 <= d_2 or d_2 + d_3 <= d_1:
        return Err, Err  # impossible triangle
    
    cos_value = (d_1**2 + d_2**2 - d_3**2) / (2 * d_1 * d_2)
    if abs(cos_value) > 1.0:
        return Err, Err  # Error for rounding
    angle_rad = math.acos(cos_value)
    return angle_rad, math.degrees(angle_rad)

def calculate_trim_points(circle_center, circle_radius, sphere_center, sphere_radius):
    cx, cy, cz = circle_center
    sx, sy, sz = sphere_center

    # Distance from sphere center to plane of circle
    dz = cz - sz
    if abs(dz) > sphere_radius:
        return []  # No intersection

    # Radius of sphere's cross-section at that plane
    r_cross = math.sqrt(sphere_radius**2 - dz**2)

    # Now we have two circles in XY plane:
    # 1. center=(cx,cy), radius=circle_radius
    # 2. center=(sx,sy), radius=r_cross
    dx = sx - cx
    dy = sy - cy
    d = math.sqrt(dx**2 + dy**2)

    # No intersection cases
    if d > circle_radius + r_cross or d < abs(circle_radius - r_cross):
        return []

    # Circle-circle intersection in 2D
    a = (circle_radius**2 - r_cross**2 + d**2) / (2*d)
    h = math.sqrt(max(circle_radius**2 - a**2, 0))
    xm = cx + a * dx / d
    ym = cy + a * dy / d

    rx = -dy * (h / d)
    ry = dx * (h / d)

    p1 = (xm + rx, ym + ry, cz)
    p2 = (xm - rx, ym - ry, cz)
    return [p1, p2]



def find_x_given_y_exclude_xb(x2, y2, yb, d6, xb):
    try:
        term = d6**2 - (yb - y2)**2
        if term < 0:
            raise ValueError("No real solution exists for the given inputs.")
        
        sqrt_term = sqrt(term)
        x5_positive = x2 + sqrt_term
        x5_negative = x2 - sqrt_term
        
        # Exclude xb from the results
        solutions = []
        if abs(x5_positive - xb) > 1e-6:  # Tolerance for floating-point comparisons
            solutions.append(x5_positive)
        if abs(x5_negative - xb) > 1e-6:
            solutions.append(x5_negative)
        
        return solutions
    except Exception as e:
        print(f"Error: {e}")
        return []
    

# Step 1: Calculate the second point
x2, y2, z2 = calculate_second_point(x1, y1, z1, d1, phi1)
print(f"Second point: ({x2:.2f}, {y2:.2f}, {z2:.2f})")

point_2 = [x2, y2, z2]
# Step 2: Calculate the distance between the second point and the base point
d2 = calculate_distance(point_2, [xb, yb, zb])
print(f"Distance between second point and base point: {d2:.2f}")

# Step 3: Calculate the angle between d3 and d4
rad_phi2, deg_phi2 = calculate_angle_cosine(d3, d4, d2)
if rad_phi2 is not None and not isinstance(rad_phi2, str):
    print(f"Angle between d3 and d4: {deg_phi2:.2f} degrees")
else:
    print(deg_phi2)
    exit()

phi_2 = deg_phi2


# #-----versition#3

# Calculate second point
x2, y2, z2 = calculate_second_point(x1, y1, z1, d1, phi1)

# Calculate d2
d2 = calculate_distance([x2, y2, z2], [xb, yb, zb])

# # Setup centers for calculations
# circle_center = np.array([x2, y2, z2])
# sphere_center = np.array([xb, yb, zb])

# Compute intersections
trim_points = calculate_trim_points((x2, y2, z2), d1, (xb, yb, zb), d2)

print("Trim points between circle and sphere:")
for pt in trim_points:
    print(f"({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})")





# Define distances and coordinates lists
x4, y4, z4 = [], [], []
d4_list, d5_list = [], []

# Distance between (x1, y1, z1) and (x2, y2, z2)
d1 = calculate_distance((x1, y1, z1), (x2, y2, z2))

print("Angle calculations for each trim point:")
for point in trim_points:
    try:
        # Trim point coordinates (already floats)
        x, y, z = point
        x4.append(x)
        y4.append(y)
        z4.append(z)
        
        # Distances from trim point
        d5_val = calculate_distance((x1, y1, z1), (x, y, z))  # Distance to (x1, y1, z1)
        d4_val = calculate_distance((x2, y2, z2), (x, y, z))  # Distance to (x2, y2, z2)
        d5_list.append(d5_val)
        d4_list.append(d4_val)
        
        # Calculate angle phi_3 using the Law of Cosines
        if d1 > 0 and d4_val > 0:  
            cos_phi3 = (d1**2 + d4_val**2 - d5_val**2) / (2 * d1 * d4_val)
            cos_phi3 = max(-1, min(1, cos_phi3))  # Clamp to [-1, 1]
            phi_3 = degrees(acos(cos_phi3))
            print(f"Trim point: ({x:.2f}, {y:.2f}, {z:.2f}), Phi_3: {phi_3:.2f}°")
        else:
            print(f"Trim point: ({x:.2f}, {y:.2f}, {z:.2f}), Phi_3: Undefined (d1 or d4 is zero)")
    except Exception as e:
        print(f"Error processing trim point {point}: {e}")


# Coordinates of first trim point (already floats)
x4, y4, z4 = trim_points[0]

# Calculate distances
d6 = calculate_distance((x2, y2), (xb, yb))  # (x2,y2) ↔ (xb,yb)
d7 = calculate_distance((xb, yb), (x4, y4))  # (xb,yb) ↔ (x4,y4)
d8 = calculate_distance((x4, y4), (x2, y2))  # (x4,y4) ↔ (x2,y2)

# Calculate angle
try:
    rad_f7_d8, angle_f7_d8 = calculate_angle_cosine(d7, d8, d6)
    print(f"Distances: d6: {d6:.2f}, d7: {d7:.2f}, d8: {d8:.2f}")
    print(f"Angle between d7 and d8: {angle_f7_d8:.2f}°")
except ValueError as e:
    print(f"Error calculating angle: {e}")

phi_4 = angle_f7_d8


# Calculate angles
try:
    rad_phi_5, angle_phi_5 = calculate_angle_cosine(d7, d6, d8)  # Angle between d7 and d6
    print(f"Distances: d6: {d6:.2f}, d7: {d7:.2f}, d8: {d8:.2f}")
    print(f"Angle between d7 and d6 (phi_5): {angle_phi_5:.2f}°")
except ValueError as e:
    print(f"Error calculating angle: {e}")



x5_solutions = find_x_given_y_exclude_xb(x2, y2, yb, d6, xb)
if x5_solutions:
    print(f"Possible x-coordinates for the point (excluding xb={xb}): {x5_solutions}")
    for x5 in x5_solutions:
        print(f"Point: ({x5:.2f}, {yb:.2f})")
else:
    print("No solution found.")



#TODO: this is an assemption. need to be addapet into diffrent [legs: axis(x,y,z)] ratios    
y5=yb


d6 = calculate_distance((x2, y2), (x5, y5))
d7 = calculate_distance((x5, y5), (xb, yb))
print(f"Distance d6: {d6:.2f}")
print(f"Distance d7: {d7:.2f}")

rad_phi_6, angle_phi_6 = calculate_angle_cosine(d6, d7, d6)
if angle_phi_6 is not None:
    print(f"Angle angle_phi_6: {angle_phi_6:.2f}°")
else:
    print("Failed to calculate angle_phi_6.")
    exit()


phi_7 = angle_phi_5 + angle_phi_6 - 60
print(f"phi_7: {phi_7:.2f}°")

phi_8 = 360 - (phi_2 + 90 + 90)
print(f"phi_8: {phi_8:.2f}°")


# Known coordinates (z9 is the z-coordinate of the first trim point)
y9, z9 = yb, trim_points[0][2]

# Calculate distances
d_yb_z2 = calculate_distance((y2, z2), (yb, zb))  # Distance (y2,z2) ↔ (yb,zb)
d_yb_z9 = calculate_distance((yb, zb), (y9, z9))  # Distance (yb,zb) ↔ (y9,z9)
d_z2_z9 = calculate_distance((y2, z2), (y9, z9))  # Distance (y2,z2) ↔ (y9,z9)

# Calculate phi_9
try:
    rad_phi_9,angle_phi_9 = calculate_angle_cosine(d_yb_z2, d_yb_z9, d_z2_z9)
    print(f"Distances: d_yb_z2: {d_yb_z2:.2f}, d_yb_z9: {d_yb_z9:.2f}, d_z2_z9: {d_z2_z9:.2f}")
    print(f"Angle between arms (angle_phi_9): {angle_phi_9:.2f}°")
except ValueError as e:
    print(f"Error calculating angle: {e}")



# Save data to CSV
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
csv_file = "test_results.csv"
file_exists = os.path.isfile(csv_file)

data = {
    'Timestamp': timestamp,
    'x1': x1, 'y1': y1, 'z1': z1,
    'x2': x2, 'y2': y2, 'z2': z2,
    'xb': xb, 'yb': yb, 'zb': zb,
    'x4': x4, 'y4': y4, 'z4': z4,
    'd1': d1, 'd2': d2, 'd3': d3, 'd4': d4,
    'd5': d5_list[0] if d5_list else None,
    'd6': d6, 'd7': d7, 'd8': d8,
    'phi1': phi1, 'phi2': phi_2, 'phi3': phi_3,
    'phi4': phi_4, 'phi5': angle_phi_5, 'phi6': angle_phi_6,
    'phi7': phi_7, 'phi8': phi_8, 'phi9': angle_phi_9
}

with open(csv_file, mode='a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=data.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(data)

print(f"\nData saved to {csv_file}")
