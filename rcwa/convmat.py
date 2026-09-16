"""Konvolüsyon (Toeplitz) matrisi — Ders 03 & Kod Dersi 01.
Periyodik bir eps(x) profilinin Fourier temsilini, mertebeleri birbirine
bağlayan bir matrise çevirir. RCWA'nın tüm 'malzeme bilgisi' buradadır.
"""
import numpy as np


def convmat1d(eps_x, P):
    """1D periyodik profilin P x P konvolüsyon matrisi.

    Parametreler
    ------------
    eps_x : 1D dizi
        Bir periyot boyunca düzgün örneklenmiş dielektrik sabiti eps(x).
    P : int
        Tutulacak uzaysal harmonik (mertebe) sayısı — tek olmalı (2M+1).

    Döndürür
    --------
    C : (P, P) karmaşık matris   (C[r,c] = eps_{r-c})
    """
    if P % 2 == 0:
        raise ValueError("P tek olmalı (2M+1).")
    Nx = eps_x.shape[0]
    # Fourier katsayıları (fftshift ile sıfır harmonik ortada)
    A = np.fft.fftshift(np.fft.fft(eps_x)) / Nx
    p0 = Nx // 2
    C = np.zeros((P, P), dtype=complex)
    for r in range(P):
        for c in range(P):
            C[r, c] = A[p0 + (r - c)]
    return C


def convmat2d(eps, Px, Py):
    """2B periyodik eps(x,y) profilinin konvolüsyon matrisi — Ders 2B & Kod 2B.

    eps : (Nx, Ny) karmaşık dizi, birim hücrede örneklenmiş dielektrik.
    Px, Py : x ve y yönünde harmonik sayıları (ikisi de tek).
    Döndürür: (Px*Py, Px*Py) matris. Mertebe sırası x-dış, y-iç (meshgrid 'ij').
    """
    Nx, Ny = eps.shape
    A = np.fft.fftshift(np.fft.fft2(eps)) / (Nx * Ny)
    p0, q0 = Nx // 2, Ny // 2
    Mx, My = Px // 2, Py // 2
    MX, MY = np.meshgrid(np.arange(-Mx, Mx + 1), np.arange(-My, My + 1), indexing='ij')
    MXf, MYf = MX.flatten(), MY.flatten()
    P = Px * Py
    C = np.zeros((P, P), dtype=complex)
    for a in range(P):
        for b in range(P):
            C[a, b] = A[p0 + (MXf[a] - MXf[b]), q0 + (MYf[a] - MYf[b])]
    return C
