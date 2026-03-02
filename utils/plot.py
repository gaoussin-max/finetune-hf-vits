import logging
import matplotlib


matplotlib.use("Agg")

MATPLOTLIB_FLAG = False


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    
    import matplotlib.pylab as plt
    import numpy as np

    # Create the alignment plot
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    # Draw the canvas to the buffer
    fig.canvas.draw()
    
    # Modern Matplotlib (3.8+) uses buffer_rgba() instead of tostring_rgb()
    # This returns an RGBA buffer (4 channels)
    canvas = fig.canvas
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    
    # Reshape to (Height, Width, 4)
    # Then slice [:, :, :3] to convert RGBA to RGB
    data = data.reshape(canvas.get_width_height()[::-1] + (4,))
    data = data[:, :, :3]
    
    plt.close(fig)
    return data
