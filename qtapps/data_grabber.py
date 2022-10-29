import skrf_qtwidgets.networkListWidget
import skrf_qtwidgets.networkPlotWidget
from skrf_qtwidgets import qt, widgets
from qtpy import QtWidgets, QtCore


class DataGrabber(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Setup UI --- #
        self.resize(825, 575)
        self.setWindowTitle("Scikit-RF Data Grabber")
        self.verticalLayout_main = QtWidgets.QVBoxLayout(self)

        self.vna_controller = widgets.VnaSelector()
        self.verticalLayout_main.addWidget(self.vna_controller)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        size_policy.setVerticalStretch(1)
        self.splitter.setSizePolicy(size_policy)

        self.measurements_widget = QtWidgets.QWidget(self.splitter)
        self.measurements_widget_layout = QtWidgets.QVBoxLayout(self.measurements_widget)
        self.measurements_widget_layout.setContentsMargins(0, 0, 0, 0)

        self.listWidget_measurements = skrf_qtwidgets.networkListWidget.NetworkListWidget(self.measurements_widget)
        self.measurement_buttons = self.listWidget_measurements.get_input_buttons()
        self.measurements_widget_layout.addWidget(self.measurement_buttons)
        self.measurements_widget_layout.addWidget(self.listWidget_measurements)

        self.save_buttons = self.listWidget_measurements.get_save_buttons()
        self.measurements_widget_layout.addWidget(self.save_buttons)

        self.ntwk_plot = skrf_qtwidgets.networkPlotWidget.NetworkPlotWidget(self.splitter)
        self.ntwk_plot.corrected_data_enabled = False

        self.verticalLayout_main.addWidget(self.splitter)
        self.splitter.setStretchFactor(1, 100)  # important that this goes at the end
        # --- END SETUP UI --- #

        self.listWidget_measurements.ntwk_plot = self.ntwk_plot
        self.listWidget_measurements.get_analyzer = self.vna_controller.get_analyzer

app = qt.single_widget_application(DataGrabber, appid="DataGrabber")
