from __future__ import annotations

import logging

import numpy as np

from CuPyNLO.util.util import to_numpy

library = "scipy"
logger = logging.getLogger(__name__)
_wrapper_depth = 0

def typewrapper(func):
    def wrapper(*args, **kwargs):
        global _wrapper_depth
        if library == "torch":
            if isinstance(args[0], np.ndarray) and func.__name__ != "interp1d":
                arr = from_numpy(args[0])
            else:
                arr = args[0]
            if func.__name__ == "fftshift" or func.__name__ == "ifftshift":
                if "axes" in kwargs:
                    axes = kwargs.pop("axes")
                    if isinstance(axes, tuple):
                        kwargs["dim"] = axes
                    else:
                        kwargs["dim"] = (axes,)
        elif library == "cupy" and isinstance(args[0], np.ndarray):
            arr = asarray(args[0])
        else:
            arr = args[0]

        _wrapper_depth += 1
        try:
            res = func(arr, *args[1:], **kwargs)
        except NameError as e:
            logger.warning(f"Defaulting to scipy due to error: {e}")
            make_scipy_default()
            res = func(arr, *args[1:], **kwargs)
        finally:
            _wrapper_depth -= 1

        if _wrapper_depth == 0:
            return to_numpy(res)
        return res
    return wrapper

def make_scipy_default():
    global _ifftshift, _ifft, _fftshift, _fft, _fftfreq, _interp1d, library
    library = "scipy"
    from scipy.fft import ifftshift as _ifftshift, ifft as _ifft, fftshift as _fftshift, fft as _fft, fftfreq as _fftfreq
    from scipy.interpolate import interp1d as _interp1d

def make_cupy_default():
    global _ifftshift, _ifft, _fftshift, _fft, _fftfreq, _interp1d, library, ndarray, asarray
    library = "cupy"
    from cupyx.scipy.fft import ifftshift as _ifftshift, ifft as _ifft, fftshift as _fftshift, fft as _fft, fftfreq as _fftfreq
    from cupyx.scipy.interpolate import interp1d as _interp1d
    from cupy import asarray, ndarray

def make_torch_default():
    global _ifftshift, _ifft, _fftshift, _fft, _fftfreq, _interp1d, library, from_numpy, Tensor
    library = "torch"
    from torch.fft import ifftshift as _ifftshift, ifft as _ifft, fftshift as _fftshift, fft as _fft, fftfreq as _fftfreq
    from torch import from_numpy, Tensor
    from scipy.interpolate import interp1d as _interp1d

@typewrapper
def FFT_t(A: np.ndarray | ndarray | Tensor, ax: int = 0):
    return ifftshift(ifft(fftshift(A, axes=(ax,)), axis=ax), axes=(ax,))

@typewrapper
def IFFT_t(A: np.ndarray | ndarray | Tensor, ax: int = 0):
    return ifftshift(fft(fftshift(A, axes=(ax,)), axis=ax), axes=(ax,))

@typewrapper
def fft(A: np.ndarray | ndarray | Tensor, axis: int = 0, **kwargs):
    if library == "torch":
        return _fft(A, dim=axis, **kwargs)
    return _fft(A, axis=axis, **kwargs)

@typewrapper
def fftshift(A: np.ndarray | ndarray | Tensor, **kwargs):
    return _fftshift(A, **kwargs)

def fftfreq(*args, **kwargs):
    return _fftfreq(*args, **kwargs)

@typewrapper
def ifft(A: np.ndarray | ndarray | Tensor, axis: int = 0, **kwargs):
    if library == "torch":
        return _ifft(A, dim=axis, **kwargs)
    return _ifft(A, axis=axis, **kwargs)

@typewrapper
def ifftshift(A: np.ndarray | ndarray | Tensor, **kwargs):
    return _ifftshift(A, **kwargs)

@typewrapper
def interp1d(x: np.ndarray | ndarray | Tensor, y: np.ndarray | ndarray, **kwargs):
    return _interp1d(x, y, **kwargs)