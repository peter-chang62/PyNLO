# -*- coding: utf-8 -*-
"""
Aliases to fast FFT implementations and associated helper functions.
"""

__all__ = ["fft", "ifft", "rfft", "irfft", "fftshift", "ifftshift", "next_fast_len"]


# %% Imports

from scipy import fft as scp_fft
import mkl_fft


# %% Definitions

#---- FFTs
def fft(x, fsc=1.0, n=None, axis=-1, overwrite_x=False):
    """
    Uses MKL to perform 1D FFT on the input array `x` along the given axis.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    fsc : float, optional
        The forward transform scale factor. The default is 1.0.
    n : int, optional
        Length of the transformed axis of the output. If `n` is smaller than
        the length of the input, the input is cropped. If it is larger, the
        input is padded with zeros. If `n` is not given, the length of the
        input along the axis specified by axis is used. The default is `None`.
    axis : int, optional
        Axis over which to compute the FFT. The default is -1.
    overwrite_x : bool, optional
        If True, the contents of x may be overwritten during the computation.
        The default is False.

    Returns
    -------
    complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by axis, or the last one if axis is not specified.
    """
    return mkl_fft.fft(x, n=n, axis=axis, overwrite_x=overwrite_x, forward_scale=fsc)

def ifft(x, fsc=1.0, n=None, axis=-1, overwrite_x=False):
    """
    Uses MKL to perform 1D IFFT on the input array `x` along the given axis.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    fsc : float, optional
        The forward transform scale factor. The reverse transform scale factor
        is set at ``1/(n*fsc)`` to yield the inverse transform. The default is
        1.0.
    n : int, optional
        Length of the transformed axis of the output. If `n` is smaller than
        the length of the input, the input is cropped. If it is larger, the
        input is padded with zeros. If `n` is not given, the length of the
        input along the axis specified by axis is used. The default is `None`.
    axis : int, optional
        Axis over which to compute the inverse FFT. The default is -1.
    overwrite_x : bool, optional
        If True, the contents of x may be overwritten during the computation.
        The default is False.

    Returns
    -------
    complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by axis, or the last one if axis is not specified.
    """
    return mkl_fft.ifft(x, n=n, axis=axis, overwrite_x=overwrite_x, forward_scale=fsc)

#---- Real FFTs
def rfft(x, fsc=1.0, n=None, axis=-1):
    """
    Uses MKL to perform 1D FFT on the real input array `x` along the given
    axis, producing complex output and giving only half of the harmonics.

    Parameters
    ----------
    x : array_like
        Input array, must be real.
    fsc : float, optional
        The forward transform scale factor. The default is 1.0.
    n : int, optional
        Number of points along transformation axis in the input to use. If `n`
        is smaller than the length of the input, the input is cropped. If it is
        larger, the input is padded with zeros. If `n` is not given, the length
        of the input along the axis specified by axis is used. The default is
        `None`.
    axis : int, optional
        Axis over which to compute the FFT. The default is -1.

    Returns
    -------
    complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by axis, or the last one if axis is not specified. If `n` is
        even, the length of the transformed axis is ``(n/2)+1``. If `n` is odd,
        the length is ``(n+1)/2``.
    """
    return mkl_fft.rfft_numpy(x, n=n, axis=axis, forward_scale=fsc)

def irfft(x, fsc=1.0, n=None, axis=-1):
    """
    Uses MKL to perform 1D IFFT on the input array `x` along the given axis,
    assumed to contain only half of the harmonics, producing real output.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    fsc : float, optional
        The forward transform scale factor. The reverse transform scale factor
        is set at ``1/(n*fsc)`` to yield the inverse transform. The default is
        1.0.
    n : int, optional
        Length of the transformed axis of the output. For `n` output points,
        ``n//2+1`` input points are necessary. If the input is longer than
        this, it is cropped. If it is shorter than this, it is padded with
        zeros. If `n` is not given, it is taken to be ``2*(m-1)``, where `m` is
        the length of the input along the axis specified by `axis`. The
        default is `None`.
    axis : int, optional
        Axis over which to compute the inverse FFT. The default is -1.

    Returns
    -------
    ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by axis, or the last one if axis is not specified. The length
        of the transformed axis is `n`, or, if `n` is not given, ``2*(m-1)``
        where `m` is the length of the transformed axis of the input. To get an
        odd number of output points, `n` must be specified.
    """
    return mkl_fft.irfft_numpy(x, n=n, axis=axis, forward_scale=fsc)

#---- Helper Functions
fftshift = scp_fft.fftshift
ifftshift = scp_fft.ifftshift
next_fast_len = scp_fft.next_fast_len