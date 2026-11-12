"""Redheffer star product — Lesson 05 & Coding Lesson 03.
Combines the scattering matrices of two layers into a single equivalent
S-matrix, accounting for the infinite internal reflections between them.
(This is NOT an ordinary matrix product.)
Each S-matrix is a 4-tuple: (S11, S12, S21, S22).
"""
import numpy as np


def redheffer_star(SA, SB):
    A11, A12, A21, A22 = SA
    B11, B12, B21, B22 = SB
    n = A11.shape[0]
    I = np.eye(n, dtype=complex)
    # D = A12 @ inv(I - B11 A22),  F = B21 @ inv(I - A22 B11)   (right-division via solve)
    D = np.linalg.solve((I - B11 @ A22).T, A12.T).T
    F = np.linalg.solve((I - A22 @ B11).T, B21.T).T
    S11 = A11 + D @ B11 @ A21
    S12 = D @ B12
    S21 = F @ A21
    S22 = B22 + F @ A22 @ B12
    return (S11, S12, S21, S22)
