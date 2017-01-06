from __future__ import print_function
import sys
import os

skrf_qtapps_path = os.path.dirname(os.path.abspath(__file__ + "/../.."))
sys.path.insert(0, skrf_qtapps_path)


if len(sys.argv) > 1:
    if sys.argv[1].lower() in ("pyqt4", "pyqt", "pyside", "pyqt5"):
        os.environ["QT_API"] = sys.argv[1].lower()
try:
    print(os.environ["QT_API"])
except KeyError:
    pass

import sip
from qtpy import QtWidgets
# from skrf_qtwidgets import qt

app = QtWidgets.QApplication(sys.argv)


def get_filename():
    filename = QtWidgets.QFileDialog.getOpenFileName()
    print(filename)

form = QtWidgets.QWidget()
layout = QtWidgets.QVBoxLayout(form)
btn_openFile = QtWidgets.QPushButton("Open File")
btn_openFile.clicked.connect(get_filename)
layout.addWidget(btn_openFile)
form.show()

sip.setdestroyonexit(False)
sys.exit(app.exec_())
