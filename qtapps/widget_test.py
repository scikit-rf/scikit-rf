import sys
from skrf_qtwidgets import qt, widgets
from qtpy import QtCore, QtWidgets

app = qt.instantiate_app(sys.argv)

class nwa_dummy():
    def __init__(self, NPORTS=2, NCHANNELS=False):
        self.NPORTS = NPORTS
        self.NCHANNELS = NCHANNELS

    def get_list_of_traces(self):
        traces = [
            {"text": "trace 1"},
            {"text": "trace 2"}
        ]
        return traces
nwa = nwa_dummy(2, 4)

form = widgets.MeasurementDialog(nwa)
form.show()

app.exec_()

