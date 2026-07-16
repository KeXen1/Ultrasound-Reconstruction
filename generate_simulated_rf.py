import os
import matplotlib.pyplot as plt
import numpy as np


num_beams = 128
samples_per_beam = 2048

fs = 40e6  # Sampling frequency: 40 MHz
fc = 5e6   # Center frequency: 5 MHz

# Make sure the output folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("images", exist_ok=True)


# -------------------------------------------------
# Step 1: Create the digital phantom
# -------------------------------------------------

phantom = np.zeros((num_beams, samples_per_beam))

# Add random weak tissue scatterers
phantom += 0.08 * np.random.randn(
    num_beams,
    samples_per_beam
)


# -------------------------------------------------
# Step 2: Add strong horizontal reflectors
# -------------------------------------------------

phantom[:, 600] += 2.0
phantom[:, 1200] += 3.0


# -------------------------------------------------
# Step 3: Add a low-reflectivity cyst
# -------------------------------------------------

center_beam = 64
center_depth = 900

beam_radius = 25
depth_radius = 120

for beam in range(num_beams):
    for depth_sample in range(samples_per_beam):

        normalized_distance = (
            ((beam - center_beam) / beam_radius) ** 2
            + ((depth_sample - center_depth) / depth_radius) ** 2
        )

        if normalized_distance < 1:
            phantom[beam, depth_sample] = 0.0


# -------------------------------------------------
# Step 4: Save the phantom image
# -------------------------------------------------

plt.figure(figsize=(8, 7))

plt.imshow(
    phantom.T,
    cmap="gray",
    aspect="auto",
    origin="upper"
)

plt.title("Digital Ultrasound Phantom")
plt.xlabel("Beam Number")
plt.ylabel("Depth Sample")
plt.colorbar(label="Reflectivity")
plt.tight_layout()

plt.savefig(
    "images/phantom.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -------------------------------------------------
# Step 5: Create a windowed ultrasound pulse
# -------------------------------------------------

time = np.arange(
    0,
    2e-6,
    1 / fs
)

pulse = np.sin(
    2 * np.pi * fc * time
)

pulse *= np.hanning(len(pulse))


# -------------------------------------------------
# Step 6: Convert the phantom into RF data
# -------------------------------------------------

rf = np.zeros_like(phantom)

for beam in range(num_beams):
    rf[beam] = np.convolve(
        phantom[beam],
        pulse,
        mode="same"
    )


# -------------------------------------------------
# Step 7: Apply depth attenuation
# -------------------------------------------------

depth = np.linspace(
    0,
    1,
    samples_per_beam
)

attenuation = np.exp(-2 * depth)

rf *= attenuation


# -------------------------------------------------
# Step 8: Add electronic noise
# -------------------------------------------------

rf += 0.02 * np.random.randn(
    num_beams,
    samples_per_beam
)


# -------------------------------------------------
# Step 9: Save one example RF signal
# -------------------------------------------------

beam_number = 64

plt.figure(figsize=(10, 4))

plt.plot(rf[beam_number])

plt.title(f"Simulated RF Signal — Beam {beam_number}")
plt.xlabel("Depth Sample")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "images/rf_signal.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -------------------------------------------------
# Step 10: Save the raw RF data
# -------------------------------------------------

np.save(
    "data/simulated_rf.npy",
    rf
)

print("RF data saved!")
print("RF shape:", rf.shape)
print("Saved images/phantom.png")
print("Saved images/rf_signal.png")