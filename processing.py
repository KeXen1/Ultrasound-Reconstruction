import numpy as np
from scipy.signal import butter, filtfilt, hilbert

def time_gain_compensation(rf):
    samples = rf.shape[1]

    depth = np.linspace(1, 3, samples)

    gain = depth**2

    return rf * gain

def bandpass_filter(rf, fs = 40e6):

    # Ultrasound probe freq range
    low = 3e6 # 3 MHz
    high = 7e6 # 7 MHz
    fn = fs/2 # Normalized freq

    # 4th-order Butterworth bandpass filter
    b, a = butter(4, [low/fn, high/fn], btype='band')

    filtered = np.zeros_like(rf)

    # Filter each beam separately
    for i in range(rf.shape[0]):
        filtered[i] = filtfilt(b, a, rf[i])
    return filtered

def envelope_detection(rf):
    envelope = np.abs(hilbert(rf, axis=1))
    return envelope

def log_compression(envelope):
    # Convert amplitude to decibels
    bmode = 20 * np.log10(envelope + 1e-6)

    # Normalize maximum intensity to 0 dB
    bmode = bmode - np.max(bmode)

    # Limit display range
    bmode = np.clip(bmode, -60, 0)

    return bmode