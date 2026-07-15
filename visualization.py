import matplotlib.pyplot as plt

def show_image(bmode):
    plt.imshow(
        bmode.T,
        cmap="gray",
        aspect="auto"
    )

    plt.xlabel("Beam")
    plt.ylabel("Depth")
    plt.title("B-mode Ultrasound")
    plt.show()