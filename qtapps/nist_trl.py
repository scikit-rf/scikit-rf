import skrf_qtwidgets.networkPlotWidget
from skrf_qtwidgets import qt, widgets, calibration_widgets
from qtpy import QtWidgets, QtCore


class NISTTRLWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(NISTTRLWidget, self).__init__(parent)

        # --- Setup UI --- #
        self.resize(950, 575)
        self.setWindowTitle("NIST Multiline TRL Calibration")
        self.verticalLayout_main = QtWidgets.QVBoxLayout(self)

        self.vna_controller = widgets.VnaController()
        self.vna_controller.verticalLayout.setContentsMargins(3, 3, 3, 3)
        self.verticalLayout_main.addWidget(self.vna_controller)
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
        self.tab_measurements.connect_plot(self.ntwk_plot)
        self.tab_measurements.get_calibration = self.tab_calStandards.get_calibration

    def closeEvent(self, event):
        quit_msg = "Are you sure you want to exit the program?"
        reply = QtWidgets.QMessageBox.question(
            self, 'Message', quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

# app = qt.single_widget_application(NISTTRLWidget, splash_screen=False)
import sys
import skrf
qt.set_process_id("skrf")
app = QtWidgets.QApplication(sys.argv)
qt.setup_style()
qt.set_popup_exceptions()

form = NISTTRLWidget()
form.show()

thru = skrf.Network(r"C:\Coding\Python\scikit-rf\qtapps\skrf_qtwidgets\example_data\Cal_S-Param\Thru.s2p")
fswitch = skrf.Network(r"C:\Coding\Python\scikit-rf\qtapps\skrf_qtwidgets\example_data\Cal_S-Param\forward.s1p")
rswitch = skrf.Network(r"C:\Coding\Python\scikit-rf\qtapps\skrf_qtwidgets\example_data\Cal_S-Param\reverse.s1p")
r1 = skrf.Network(r"C:\Coding\Python\scikit-rf\qtapps\skrf_qtwidgets\example_data\Cal_S-Param\Short1.s1p")
r2 = skrf.Network(r"C:\Coding\Python\scikit-rf\qtapps\skrf_qtwidgets\example_data\Cal_S-Param\Short2.s1p")
reflect = skrf.two_port_reflect(r1, r2)
l1 = skrf.Network(r"C:\Coding\Python\scikit-rf\qtapps\skrf_qtwidgets\example_data\Cal_S-Param\L1.s2p")
l2 = skrf.Network(r"C:\Coding\Python\scikit-rf\qtapps\skrf_qtwidgets\example_data\Cal_S-Param\L2.s2p")
l3 = skrf.Network(r"C:\Coding\Python\scikit-rf\qtapps\skrf_qtwidgets\example_data\Cal_S-Param\L3.s2p")
l4 = skrf.Network(r"C:\Coding\Python\scikit-rf\qtapps\skrf_qtwidgets\example_data\Cal_S-Param\L4.s2p")

form.tab_calStandards.listWidget_thru.load_named_ntwk(thru, form.tab_calStandards.THRU_ID, False)
form.tab_calStandards.listWidget_thru.load_named_ntwk(fswitch, form.tab_calStandards.SWITCH_TERMS_ID_FORWARD, False)
form.tab_calStandards.listWidget_thru.load_named_ntwk(rswitch, form.tab_calStandards.SWITCH_TERMS_ID_REVERSE, False)
form.tab_calStandards.listWidget_reflect.load_network(reflect, False)

# form.tab_calStandards.listWidget_line.load_networks((l1, l2, l3, l4))
form.tab_calStandards.listWidget_line.load_network(l1, parameters={"length": 0.280}, activate=False)
form.tab_calStandards.listWidget_line.load_network(l2, parameters={"length": 0.158}, activate=False)
form.tab_calStandards.listWidget_line.load_network(l3, parameters={"length": 0.105}, activate=False)
form.tab_calStandards.listWidget_line.load_network(l4, parameters={"length": 0.070}, activate=False)
form.tab_measurements.listWidget_measurements.load_networks((l1, l2, l3, l4))

qt.sip.setdestroyonexit(False)  # prevent a crash on exit
sys.exit(app.exec_())
