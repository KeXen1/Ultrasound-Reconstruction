import numpy as np

num_beams = 128
samples_per_beam = 2048

fs = 40e6 # Sampling freq
fc = 5e6 # Center freq

# Create phantom
phantom = np.zeros((num_beams, samples_per_beam))

# Random weak scatters
phantom += 0.08 * np.random.randn(num_beams, samples_per_beam)

# Add simulated echo reflections
for beam in range(num_beams):
    phantom[beam, 600] += 2.0
    phantom[beam, 1200] += 3.0

# Circular Cyst
center_beam = 64
center_depth = 900

beam_radius = 25
depth_radius = 120

for beam in range(num_beams):
    for depth in range(samples_per_beam):

        normalized_distance = (
            ((beam - center_beam) / beam_radius) ** 2
            + ((depth - center_depth) / depth_radius) ** 2
        )

        if normalized_distance < 1:
            phantom[beam, depth] = 0.0

# Create Ultrasound Pulse

time = np.arange(0, 2e-6, 1/fs)

pulse = np.sin(2 * np.pi * fc * time)

pulse *= np.hanning(len(pulse))

# Convert phantom to RF

rf = np.zeros_like(phantom)

for beam in range(num_beams):
    rf[beam] = np.convolve(
        phantom[beam],
        pulse,
        mode="same"
    )

# Add Depth Attenuation
depth = np.linspace(0, 1, samples_per_beam)
attenuation = np.exp(-2 * depth)
rf *= attenuation

# Electronic Noise
rf += 0.02 * np.random.randn(num_beams, samples_per_beam)

# Save raw RF data
np.save("data/simulated_rf.npy", rf)

print("RF data saved!")