import io
import matplotlib.pyplot as plt

try:
    from PyQt5.QtGui import QGuiApplication, QImage
except:
    from PyQt6.QtGui import QGuiApplication, QImage
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
import matplotlib


def add_clipboard_to_figures():
    # use monkey-patching to replace the original plt.figure() function with
    # our own, which supports clipboard-copying
    oldfig = plt.figure

    def newfig(*args, **kwargs):
        fig = oldfig(*args, **kwargs)

        def clipboard_handler(event):
            if event.key == "cmd+c":
                # store the image in a buffer using savefig(), this has the
                # advantage of applying all the default savefig parameters
                # such as background color; those would be ignored if you simply
                # grab the canvas using Qt
                buf = io.BytesIO()
                fig.savefig(buf, transparent=True, dpi=300)
                QGuiApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))
                buf.close()

        fig.canvas.mpl_connect("key_press_event", clipboard_handler)
        return fig

    plt.figure = newfig


add_clipboard_to_figures()

# -----------------------------------------------------------------------------
# add a transparent background color map that is the CMRmap with white -> transparent
ncolors = 256
color_array = plt.get_cmap("CMRmap")(range(ncolors))

# change alpha values
color_array[-1][-1] = 0  # just send the white values to transparent!

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name="CMRmap_t", colors=color_array)

# register this new colormap with matplotlib
colormaps.register(cmap=map_object)

# -----------------------------------------------------------------------------
# add a transparent background color map that is the CMRmap_r with white -> transparent
ncolors = 256
color_array = plt.get_cmap("CMRmap_r")(range(ncolors))

# change alpha values
color_array[0][-1] = 0  # just send the white values to transparent!

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name="CMRmap_r_t", colors=color_array)

# register this new colormap with matplotlib
colormaps.register(cmap=map_object)

# -----------------------------------------------------------------------------
# add a transparent background color map that is the binary with white -> transparent
ncolors = 256
color_array = plt.get_cmap("binary")(range(ncolors))

# change alpha values
color_array[0][-1] = 0  # just send the white values to transparent!

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name="binary_t", colors=color_array)

# register this new colormap with matplotlib
colormaps.register(cmap=map_object)

# -----------------------------------------------------------------------------
# add a transparent background color map that is the binary with white -> transparent
ncolors = 256
color_array = plt.get_cmap("gnuplot2_r")(range(ncolors))

# change alpha values
color_array[0][-1] = 0  # just send the white values to transparent!

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name="gnuplot2_r_t", colors=color_array)

# register this new colormap with matplotlib
colormaps.register(cmap=map_object)

# -----------------------------------------------------------------------------
matplotlib.rc("image", cmap="RdBu_r")
