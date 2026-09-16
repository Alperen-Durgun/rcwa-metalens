"""Redheffer star product — Ders 05 & Kod Dersi 03.
İki katmanın scattering matrisini, aralarındaki sonsuz iç yansımaları
hesaba katarak tek bir eşdeğer S-matrise birleştirir. (Düz çarpım DEĞİL.)
Her S-matris bir 4'lü demettir: (S11, S12, S21, S22).
"""
import numpy as np


def redheffer_star(SA, SB):
    A11, A12, A21, A22 = SA
    B11, B12, B21, B22 = SB
    n = A11.shape[0]
    I = np.eye(n, dtype=complex)
    D = A12 @ np.linalg.inv(I - B11 @ A22)
    F = B21 @ np.linalg.inv(I - A22 @ B11)
    S11 = A11 + D @ B11 @ A21
    S12 = D @ B12
    S21 = F @ A21
    S22 = B22 + F @ A22 @ B12
    return (S11, S12, S21, S22)
