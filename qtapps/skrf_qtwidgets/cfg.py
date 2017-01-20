import os
import sys

this_dir = os.path.normpath(os.path.dirname(__file__))
images_dir = os.path.join(this_dir, "images/")
example_data_dir = os.path.join(this_dir, "example_data/")
executable_dir = os.getcwd()
user_dir = os.path.expanduser("~")

last_path = os.path.join(this_dir, "example_data/")

if not os.path.isdir(last_path):
    last_path = user_dir

path_default = last_path
skrf_icon = os.path.join(images_dir, "scikit-rf-logo.png")

os.environ['SKRF_PLOT_ENV'] = "none"  # disable automatic loading of pylab

os.environ['QT_API'] = 'pyqt5'  # force prefer pyqt5, let qtpy handle pyqt4 or pyside only
if len(sys.argv) > 1:
    if sys.argv[1].lower() in ("pyqt4", "pyqt", "pyside", "pyqt5"):
        os.environ["QT_API"] = sys.argv[1].lower()

preferred_styles = ['plastique', 'Fusion', 'cleanlooks', 'motif', 'cde']
preferred_style = 'plastique'
