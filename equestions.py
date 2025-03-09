import numpy as np
from sympy import symbols, Eq, solve
import math
from math import sqrt, acos, degrees

# Define known variables
x1, y1, z1 = 1.0, 2.0, 3.0  # Example coordinates of the first point
d1 = 5.0  # Distance from the first point to the second
phi1 = 45.0  # Angle in degrees clockwise
xb, yb, zb = 7.0, 8.0, 9.0  # Coordinates of the base point
d3 = 6.0  # Example value for distance d3
d4 = 7.0  # Example value for distance d4


def calculate_second_point(x1, y1, z1, d1, phi1):
    # Convert phi1 to radians for calculations
    phi1_rad = math.radians(phi1)
    x2 = x1 + d1 * math.cos(phi1_rad)
    y2 = y1 + d1 * math.sin(phi1_rad)
    z2 = z1  # Assuming z2 remains the same since no vertical angle is provided
    return x2, y2, z2

def calculate_angle_between_vectors(d3, d4, d2):
    # Using the cosine rule
    try:
        angle_rad = math.acos((d3 ** 2 + d4 ** 2 - d2 ** 2) / (2 * d3 * d4))
        angle_deg = math.degrees(angle_rad)
        return angle_rad, angle_deg
    except ValueError:
        return None, "Invalid input values for calculating angle. Check distances."


def calculate_second_point(x1, y1, z1, d1, phi1):
    # Convert phi1 to radians for calculations
    phi1_rad = math.radians(phi1)
    x2 = x1 + d1 * math.cos(phi1_rad)
    y2 = y1 + d1 * math.sin(phi1_rad)
    z2 = z1  # Assuming z2 remains the same since no vertical angle is provided
    return x2, y2, z2

def calculate_angle_between_vectors(d3, d4, d2):
    # Using the cosine rule
    try:
        angle_rad = math.acos((d3 ** 2 + d4 ** 2 - d2 ** 2) / (2 * d3 * d4))
        angle_deg = math.degrees(angle_rad)
        return angle_rad, angle_deg
    except ValueError:
        return None, "Invalid input values for calculating angle. Check distances."

def get_circle_points(center, radius, num_points=100):
    # Generate points on a circle in 3D space
    t = np.linspace(0, 2*np.pi, num_points)
    x = np.full_like(t, center[0])  # Circle is in YZ plane
    y = center[1] + radius * np.cos(t)
    z = center[2] + radius * np.sin(t)
    return np.column_stack((x, y, z))

def get_sphere_points(center, radius, num_points=1000):
    # Generate points on a sphere
    phi = np.linspace(0, np.pi, int(np.sqrt(num_points)))
    theta = np.linspace(0, 2*np.pi, int(np.sqrt(num_points)))
    phi, theta = np.meshgrid(phi, theta)
    
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)
    return np.column_stack((x.flatten(), y.flatten(), z.flatten()))

def find_intersection_points(circle_points, sphere_points, tolerance=0.1):
    # Find intersection points between circle and sphere
    intersection_points = []
    for cp in circle_points:
        # Find closest sphere points
        distances = np.linalg.norm(sphere_points - cp, axis=1)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] < tolerance:
            intersection_points.append(cp)
    return np.array(intersection_points)

def calculate_trim_points(circle_center, circle_radius, sphere_center, sphere_radius):
    # ! BUG : To calculate the eq. menualy - Hedva 2 ???
    # Define symbolic variables for the equations
    X, Y, Z = symbols('X Y Z')

    # Sphere equation: (X - xb)^2 + (Y - yb)^2 + (Z - zb)^2 = sphere_radius^2
    sphere_eq = Eq((X - sphere_center[0])**2 + (Y - sphere_center[1])**2 + (Z - sphere_center[2])**2, sphere_radius**2)

    # Circle equation in 3D: (X - x2)^2 + (Y - y2)^2 = circle_radius^2 and Z = z2
    circle_eq = Eq((X - circle_center[0])**2 + (Y - circle_center[1])**2, circle_radius**2)
    plane_eq = Eq(Z, circle_center[2])

    # Solve the system of equations
    solutions = solve([sphere_eq, circle_eq, plane_eq], (X, Y, Z))
    return solutions

def calculate_distance(point1, point2):
        return sqrt((point1[0] - point2[0])**2 + 
                (point1[1] - point2[1])**2 + 
                (point1[2] - point2[2])**2)


# Function to calculate 2D distance
def calculate_distance_2d(point1, point2):
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to calculate the angle using the cosine law
def calculate_angle_cosine(d1, d2, d3):
    # Ensure distances are valid for a triangle
    if d1 * d2 == 0:
        raise ValueError("Invalid distances, cannot calculate angle.")
    # Apply the cosine law formula
    cos_theta = (d1**2 + d2**2 - d3**2) / (2 * d1 * d2)
    cos_theta = max(min(cos_theta, 1), -1)  # Clamp to avoid numerical issues
    return degrees(acos(cos_theta))


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


# Function to calculate phi_6
def calculate_phi_6(x2, y2, x5, y5, xb, yb):
    try:
        # Calculate distances
        d6 = calculate_distance_2d((x2, y2), (x5, y5))
        d7 = calculate_distance_2d((x5, y5), (xb, yb))
        d8 = calculate_distance_2d((x2, y2), (xb, yb))
        
        # Use the Law of Cosines to calculate the angle
        if d6 > 0 and d7 > 0 and d8 > 0:  # Ensure distances are non-zero
            cos_phi_6 = (d6**2 + d8**2 - d7**2) / (2 * d6 * d8)
            # Clamp cos_phi_6 to the range [-1, 1] to handle floating-point inaccuracies
            cos_phi_6 = max(-1, min(1, cos_phi_6))
            phi_6 = degrees(acos(cos_phi_6))
            return d6, d7, phi_6
        else:
            raise ValueError("One or more distances are zero, cannot calculate angle.")
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

# Function to calculate distance in 2D
def calculate_distance_2d(coord1, coord2):
    return sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

# Function to calculate angle using cosine law
def calculate_angle_cosine(d1, d2, d3):
    if d1 * d2 == 0:
        raise ValueError("Invalid distances, cannot calculate angle.")
    cos_theta = (d1**2 + d2**2 - d3**2) / (2 * d1 * d2)
    cos_theta = max(min(cos_theta, 1), -1)  # Clamp to avoid numerical issues
    return degrees(acos(cos_theta))


# Step 1: Calculate the second point
x2, y2, z2 = calculate_second_point(x1, y1, z1, d1, phi1)
print(f"Second point: ({x2:.2f}, {y2:.2f}, {z2:.2f})")

# Step 2: Calculate the distance between the second point and the base point
d2 = calculate_distance((x2, y2, z2), (xb, yb, zb))
print(f"Distance between second point and base point: {d2:.2f}")

# Step 3: Calculate the angle between d3 and d4
angle_rad, angle_deg = calculate_angle_between_vectors(d3, d4, d2)
if angle_rad is not None:
    print(f"Angle between d3 and d4: {angle_deg:.2f} degrees")
else:
    print(angle_deg)

phi_2 = angle_deg


# Step 3: Generate circle points centered at (x2, y2, z2) with radius d1
circle_center = np.array([x2, y2, z2])
circle_points = get_circle_points(circle_center, d1)
print("Circle equation: (x - {})² + (y - {})² + (z - {})² = {}²".format(x2, y2, z2, d1))

# Step 4: Generate sphere points centered at (xb, yb, zb) with radius d2
sphere_center = np.array([xb, yb, zb])
sphere_points = get_sphere_points(sphere_center, d2)
print("Sphere equation: (x - {})² + (y - {})² + (z - {})² = {}²".format(xb, yb, zb, d2))

# Step 5: Find intersection points (trim points)
# trim_points = find_intersection_points(circle_points, sphere_points)
trim_points_complex = []
trim_points = []
solutions = calculate_trim_points(circle_center, d1, sphere_center, d2)
for point in solutions:
    # Check if any coordinate in the point contains 'I'
    if any('I' in str(coord) for coord in point):
        trim_points_complex.append(point)
    else:
        trim_points.append(point)

print("Trim points between circle and sphere:")
for point in trim_points:
    try:
        # Attempt to evaluate and format each coordinate
        x_trim, y_trim, z_trim = [float(coord.evalf()) for coord in point]
        print(f"({x_trim:.2f}, {y_trim:.2f}, {z_trim:.2f})")
    except (TypeError, ValueError):
        # Handle cases where the evaluation fails
        print(f"Could not evaluate point: {point}")

print("\nComplex trim points:")
for point in trim_points_complex:
    try:
        # Print the complex points directly since they can't be converted to float
        print(f"({point[0]}, {point[1]}, {point[2]})")
    except (TypeError, ValueError):
        print(f"Could not format complex point: {point}")

# Define distances and coordinates lists
x4, y4, z4 = [], [], []
d4, d5 = [], []


# Distance between (x1, y1, z1) and (x2, y2, z2)
d1 = calculate_distance((x1, y1, z1), (x2, y2, z2))

print("Angle calculations for each trim point:")
for point in trim_points:
    try:
        # Extract trim point coordinates
        x, y, z = [float(coord.evalf()) for coord in point]
        x4.append(x)
        y4.append(y)
        z4.append(z)
        
        # Calculate distances
        d5_val = calculate_distance((x1, y1, z1), (x, y, z))  # Distance to (x1, y1, z1)
        d4_val = calculate_distance((x2, y2, z2), (x, y, z))  # Distance to (x2, y2, z2)
        d5.append(d5_val)
        d4.append(d4_val)
        
        # Calculate angle phi_3 using the Law of Cosines
        # cos(phi_3) = (d1^2 + d4[i]^2 - d5[i]^2) / (2 * d1 * d4[i])
        if d1 > 0 and d4_val > 0:  # Ensure distances are non-zero to avoid division errors
            cos_phi3 = (d1**2 + d4_val**2 - d5_val**2) / (2 * d1 * d4_val)
            # Clamp cos_phi3 to the range [-1, 1] to avoid math domain errors due to floating-point inaccuracies
            cos_phi3 = max(-1, min(1, cos_phi3))
            phi_3 = degrees(acos(cos_phi3))  # Convert to degrees
            print(f"Trim point: ({x:.2f}, {y:.2f}, {z:.2f}), Phi_3: {phi_3:.2f}°")
        else:
            print(f"Trim point: ({x:.2f}, {y:.2f}, {z:.2f}), Phi_3: Undefined (d1 or d4 is zero)")
    except Exception as e:
        print(f"Error processing trim point {point}: {e}")


# Coordinates of points
x4, y4, z4 = [float(coord.evalf()) for coord in trim_points[0]]  # First trim point

# Calculate distances
d6 = calculate_distance_2d((x2, y2), (xb, yb))  # Distance between (x2, y2) and (xb, yb)
d7 = calculate_distance_2d((xb, yb), (x4, y4))  # Distance between (xb, yb) and (x4, y4)
d8 = calculate_distance_2d((x4, y4), (x2, y2))  # Distance between (x4, y4) and (x2, y2)

# Calculate angle between d7 and d8 using cosine law
try:
    angle_f7_d8 = calculate_angle_cosine(d7, d8, d6)
    print(f"Distances: d6: {d6:.2f}, d7: {d7:.2f}, d8: {d8:.2f}")
    print(f"Angle between d7 and d8: {angle_f7_d8:.2f}°")
except ValueError as e:
    print(f"Error calculating angle: {e}")

phi_4 = angle_f7_d8 


# Calculate angles
try:
    phi_5 = calculate_angle_cosine(d7, d6, d8)  # Angle between d7 and d6
    print(f"Distances: d6: {d6:.2f}, d7: {d7:.2f}, d8: {d8:.2f}")
    print(f"Angle between d7 and d6 (phi_5): {phi_5:.2f}°")
except ValueError as e:
    print(f"Error calculating angle: {e}")

x5_solutions = find_x_given_y_exclude_xb(x2, y2, yb, d6, xb)
x5, y5, z5 = x5_solutions[0], yb, zb  # Take the first solution for x5
if x5_solutions:
    print(f"Possible x-coordinates for the point (excluding xb={xb}): {x5_solutions}")
    print(f"Selected point: ({x5:.2f}, {yb:.2f})")
else:
    print("No solution found.")


d6, d7, phi_6 = calculate_phi_6(x2, y2, x5, y5, xb, yb)
if phi_6 is not None:
    print(f"Distance d6: {d6:.2f}")
    print(f"Distance d7: {d7:.2f}")
    print(f"Angle phi_6: {phi_6:.2f}°")
else:
    print("Failed to calculate phi_6.")


phi_7 = phi_5 + phi_6 - 60
phi_7

phi_8 = 360 - (phi_2 + 90 + 90)
print(f"phi_8: {phi_8:.2f}°")


# Known coordinates
y9, z9 = yb, float(trim_points[0][2].evalf())  # Coordinates of (y9, z9), z9 = z4

# Calculate distances
d_yb_z2 = calculate_distance_2d((y2, z2), (yb, zb))  # Distance between (y2, z2) and (yb, zb)
d_yb_z9 = calculate_distance_2d((yb, zb), (y9, z9))  # Distance between (yb, zb) and (y9, z9)
d_z2_z9 = calculate_distance_2d((y2, z2), (y9, z9))  # Distance between (y2, z2) and (y9, z9)

# Calculate phi_9
try:
    phi_9 = calculate_angle_cosine(d_yb_z2, d_yb_z9, d_z2_z9)  # Angle between arms
    print(f"Distances: d_yb_z2: {d_yb_z2:.2f}, d_yb_z9: {d_yb_z9:.2f}, d_z2_z9: {d_z2_z9:.2f}")
    print(f"Angle between arms (phi_9): {phi_9:.2f}°")
except ValueError as e:
    print(f"Error calculating angle: {e}")




