"""
[requirement]
* scipy
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from scipy.fftpack import rfft, rfftfreq, irfft



def zero_padding(data, axis=-1, pad=1):
    dim = len(data.shape)
    data_len = data.shape[axis]
    pad_len = int(data_len * (pad-1))
    axis = dim - 1 if axis == -1 else axis

    padding_index = []
    for i in range(dim):
        padding_index.append((0, pad_len if i == axis else 0))
    padded = np.pad(data, padding_index, 'constant', constant_values=0)
    return padded


def unpadding(padded_data, shape):
    slices = []
    for i in shape:
        slices.append(slice(0, i))
    slices = tuple(slices)
    return padded_data[slices]


def target_axis_to_rear(data, axis):
    if axis == -1:
        transposed = data
        transpose_shape = tuple(range(len(data.shape)))
    else:
        transpose_shape = []
        last_index = len(data.shape)-1
        for i, s in enumerate(data.shape):
            if i == axis:
                transpose_shape.append(last_index)
                last_index = i
            else:
                transpose_shape.append(i)
        transpose_shape[-1] = last_index
        transpose_shape = tuple(transpose_shape)
        transposed = np.transpose(data, transpose_shape)
    return transposed, transpose_shape


def get_fft(data, axis=-1, pad=1):
    padded = zero_padding(data, axis, pad) if pad > 1 else data
    fft = rfft(padded)
    return fft


def get_fft_freq(length, fps=None, time_step=None):
    if fps is None and time_step is None:
        print("Either fps or time_step must not be None.")
        return None
    time_step =  time_step if fps is None else (1.0 / fps)
    freq = rfftfreq(length, time_step)
    return freq


def get_ifft(data, axis=-1):
    signal = irfft(data, axis=axis)
    return signal


def get_frequency_band_mask(freq, band=(None, None)):
    if band[0] is None and band[1] is None:
        return np.zeros_like(freq).astype(np.bool)
    elif band[0] is not None and band[1] is not None:
        low_band, high_band = band
        if low_band > high_band:
            return (freq < low_band) & (freq > high_band)
        else:
            return (freq < low_band) | (freq > high_band)
    elif band[0] is None:
        high_band = band[1]
        return freq < high_band
    else:
        low_band = band[0]
        return freq > low_band


def band_amplification_filtering(data, coef, band=(None, None), fps=None, time_step=None, axis=-1, pad=1):
    # Data normalize
    data_mean = data.mean(axis=axis)
    normed = data - data_mean

    # Transpose for FFT
    transposed, transpose_shape = target_axis_to_rear(normed, axis)
    axis = -1

    # FFT
    fft = get_fft(transposed, axis, pad)

    # Create mask for frequency filtering
    freq = get_fft_freq(fft.shape[axis], fps, time_step)
    mask = get_frequency_band_mask(freq, band)

    # Amplication
    fft[..., mask==False] *= coef

    # Inverse FFT
    filtered = get_ifft(fft, axis)
    unpadded = unpadding(filtered, transposed.shape)
    restored = np.transpose(unpadded, transpose_shape)

    # Recover data and clipping
    clipped = np.clip(restored + data_mean, 0, 255)

    return clipped.astype(np.uint8)


def band_pass_filtering(data, band=(None, None), fps=None, time_step=None, axis=-1, pad=1):
    # Data normalize
    data_mean = data.mean(axis=axis)
    normed = data - data_mean

    # Transpose for
    transposed, transpose_shape = target_axis_to_rear(normed, axis)
    axis = -1

    fft = get_fft(transposed, axis, pad)
    freq = get_fft_freq(fft.shape[axis], fps, time_step)
    if freq is None:
        return

    mask = get_frequency_band_mask(freq, band)
    fft[..., mask] = 0

    filtered = get_ifft(fft, axis)

    unpadded = unpadding(filtered, transposed.shape)
    restored = np.transpose(unpadded, transpose_shape)
    clipped = np.clip(restored + data_mean, 0, 255)

    return clipped.astype(np.uint8)