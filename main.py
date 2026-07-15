import numpy as np
import matplotlib.pyplot as plt
from processing import (
    time_gain_compensation,
    bandpass_filter,
    envelope_detection,
    log_compression
)
from visualization import show_image

rf = np.load("data/simulated_rf.npy")

print("Original RF shape:", rf.shape)

# Processing Pipeline
rf_tgc = time_gain_compensation(rf)
rf_bpf = bandpass_filter(rf_tgc)
envelope = envelope_detection(rf_bpf)
bmode = log_compression(envelope)

print("B-mode shape:", bmode.shape)

# Display Image
show_image(bmode)