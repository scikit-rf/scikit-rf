import skrf_qtwidgets.networkPlotWidget
from skrf_qtwidgets import qt, widgets, calibration_widgets
from qtpy import QtWidgets, QtCore


class TRLWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Setup UI --- #
        self.resize(825, 575)
        self.setWindowTitle("Multiline TRL Calibration")
        self.verticalLayout_main = QtWidgets.QVBoxLayout(self)

        self.vna_controller = widgets.VnaSelector()
        self.vna_controller.verticalLayout.setContentsMargins(3, 3, 3, 3)
        self.verticalLayout_main.addWidget(self.vna_controller)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        size_policy.setVerticalStretch(1)
        self.splitter.setSizePolicy(size_policy)

        self.tabWidget = QtWidgets.QTabWidget(self.splitter)
        self.tab_calStandards = calibration_widgets.TRLStandardsWidget()
        self.tab_measurements = calibration_widgets.CalibratedMeasurementsWidget()
        self.tabWidget.addTab(self.tab_calStandards, "Cal Standards")
        self.tabWidget.addTab(self.tab_measurements, "Measurements")

        self.ntwk_plot = skrf_qtwidgets.networkPlotWidget.NetworkPlotWidget(self.splitter)

        self.verticalLayout_main.addWidget(self.splitter)
        self.splitter.setStretchFactor(1, 100)
        # --- END SETUP UI --- #

        # necessary rigging of widgets
        self.tab_calStandards.connect_plot(self.ntwk_plot)
        self.tab_measurements.connect_plot(self.ntwk_plot)
        self.tab_measurements.get_calibration = self.tab_calStandards.get_calibration

app = qt.single_widget_application(TRLWidget)
