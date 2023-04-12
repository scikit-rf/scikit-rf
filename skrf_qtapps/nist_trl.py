import skrf_qtwidgets.networkPlotWidget
from skrf_qtwidgets import qt, widgets, calibration_widgets
from qtpy import QtWidgets, QtCore


class NISTTRLWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Setup UI --- #
        self.resize(950, 575)
        self.setWindowTitle("NIST Multiline TRL Calibration")
        self.verticalLayout_main = QtWidgets.QVBoxLayout(self)

        self.vna_selector = widgets.VnaSelector()
        self.vna_selector.verticalLayout.setContentsMargins(3, 3, 3, 3)
        self.verticalLayout_main.addWidget(self.vna_selector)
        self.verticalLayout_main.addWidget(widgets.qt.QHLine())

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        size_policy.setVerticalStretch(1)
        self.splitter.setSizePolicy(size_policy)

        self.tabWidget = QtWidgets.QTabWidget(self.splitter)
        self.tab_calStandards = calibration_widgets.NISTTRLStandardsWidget()
        self.tab_measurements = calibration_widgets.CalibratedMeasurementsWidget()
        self.tabWidget.addTab(self.tab_calStandards, "Cal Standards")
        self.tabWidget.addTab(self.tab_measurements, "Measurements")

        self.ntwk_plot = skrf_qtwidgets.networkPlotWidget.NetworkPlotWidget(self.splitter)

        self.verticalLayout_main.addWidget(self.splitter)
        self.splitter.setStretchFactor(1, 100)
        # --- END SETUP UI --- #

        # necessary rigging of widgets
        self.tab_calStandards.connect_plot(self.ntwk_plot)
        self.tab_calStandards.get_analyzer = self.vna_selector.get_analyzer
        self.tab_measurements.connect_plot(self.ntwk_plot)
        self.tab_measurements.get_calibration = self.tab_calStandards.get_calibration
        self.tab_measurements.get_analyzer = self.vna_selector.get_analyzer
        self.vna_selector.enableStateToggled.connect(self.process_vna_available)
        self.tab_calStandards.calibration_updated.connect(self.tab_measurements.set_calibration)

        self.process_vna_available(self.vna_selector.isEnabled())

    def process_vna_available(self, available):
        """
        Parameters
        ----------
        available : bool
            set the widgets based on whether or not analyzers are available for measuring / setting properties
        """
        self.tab_calStandards.process_vna_available(available)
        self.tab_measurements.btn_measureMeasurement.setEnabled(available)

    def closeEvent(self, event):
        quit_msg = "Are you sure you want to exit the program?"
        reply = QtWidgets.QMessageBox.question(
            self, 'Message', quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = qt.single_widget_application(NISTTRLWidget, splash_screen=True)
