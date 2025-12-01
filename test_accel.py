import numpy as np
from accel import get_accel

G = 6.67259e-8

print("Testing C++ accel module")
print("-" * 50)

N = 3
X = np.array([
    [0.0, 0.0, 0.0],
    [1.0e13, 0.0, 0.0],
    [0.0, 2.0e13, 0.0]
], dtype=np.float64)

M = np.array([1.989e33, 5.972e27, 5.972e27], dtype=np.float64)

print(f"Number of bodies: {N}")
print(f"Positions shape: {X.shape}")
print(f"Masses shape: {M.shape}")

A = get_accel(X, M)

print(f"\nAccelerations shape: {A.shape}")
print(f"Accelerations:\n{A}")

r01 = np.linalg.norm(X[0] - X[1])
expected_a1 = -G * M[0] / r01**2
print(f"\nExpected x-acceleration for body 1: {expected_a1:.6e} cm/s^2")
print(f"Computed x-acceleration for body 1: {A[1,0]:.6e} cm/s^2")
print(f"Relative error: {abs((A[1,0] - expected_a1)/expected_a1):.6e}")

print("\nC++ module test complete")