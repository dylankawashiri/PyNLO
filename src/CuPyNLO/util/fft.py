from __future__ import annotations

import numpy as np


library = "scipy"

def make_scipy_default():
    global ifftshift, ifft, fftshift, fft, library
    library = "scipy"
    from scipy.fft import ifftshift, ifft, fftshift, fft

def make_cupy_default():
    global ifftshift, ifft, fftshift, fft, library
    library = "cupy"
    from cupyx.scipy.fft import ifftshift, ifft, fftshift, fft

def make_torch_default():
    global ifftshift, ifft, fftshift, fft, library
    library = "torch"
    from torch.fft import ifftshift, ifft, fftshift, fft

def FFT_t(A: np.ndarray, ax: int = 0):
    if library == "torch":
        A = torch.from_numpy(A)
    return ifftshift(ifft(fftshift(A,axes=(ax,)),axis=ax),axes=(ax,))

def IFFT_t(A: np.ndarray, ax: int = 0):
    if library == "torch":
        A = torch.from_numpy(A)
    return ifftshift(fft(fftshift(A,axes=(ax,)),axis=ax),axes=(ax,))

