import os

import matplotlib.pyplot as plt
import numpy as np

from processing import (
    time_gain_compensation,
    bandpass_filter,
    envelope_detection,
    log_compression
)

from visualization import show_image


# Make sure the images folder exists
os.makedirs("images", exist_ok=True)


# -------------------------------------------------
# Step 1: Load the simulated RF data
# -------------------------------------------------

rf = np.load(
    "data/simulated_rf.npy"
)

print("Original RF shape:", rf.shape)


# -------------------------------------------------
# Step 2: Run the processing pipeline
# -------------------------------------------------

rf_tgc = time_gain_compensation(rf)

rf_bpf = bandpass_filter(rf_tgc)

envelope = envelope_detection(rf_bpf)

bmode = log_compression(envelope)

print("B-mode shape:", bmode.shape)


# -------------------------------------------------
# Step 3: Save the filtered RF and envelope plot
# -------------------------------------------------

beam_number = 64

plt.figure(figsize=(10, 5))

plt.plot(
    rf_bpf[beam_number],
    label="Filtered RF Signal",
    linewidth=1
)

plt.plot(
    envelope[beam_number],
    label="Detected Envelope",
    linewidth=2
)

plt.title(
    f"Envelope Detection — Beam {beam_number}"
)

plt.xlabel("Depth Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "images/envelope.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -------------------------------------------------
# Step 4: Save the final B-mode image
# -------------------------------------------------

plt.figure(figsize=(8, 8))

plt.imshow(
    bmode.T,
    cmap="gray",
    aspect="auto",
    origin="upper",
    vmin=-60,
    vmax=0
)

plt.title("B-mode Ultrasound Reconstruction")
plt.xlabel("Beam Number")
plt.ylabel("Depth Sample")
plt.colorbar(label="Amplitude (dB)")
plt.tight_layout()

plt.savefig(
    "images/bmode.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -------------------------------------------------
# Step 5: Display the final image
# -------------------------------------------------

show_image(bmode)

print("Saved images/envelope.png")
print("Saved images/bmode.png")