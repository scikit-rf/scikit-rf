from qtpy import QtWidgets, QtCore
import skrf

from . import qt
from . import widgets


class CalibratedMeasurementsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(CalibratedMeasurementsWidget, self).__init__(parent)

        self.verticalLayout_main = QtWidgets.QVBoxLayout(self)
        self.verticalLayout_main.setContentsMargins(6, 6, 6, 6)

        self.listWidget_measurements = widgets.NetworkListWidget(self)
        self.btn_measureMeasurement = self.listWidget_measurements.get_measure_button()
        self.btn_loadMeasurement = self.listWidget_measurements.get_load_button()
        self.btn_calibrate = QtWidgets.QPushButton("Calibrate")
        self.horizontalLayout_measurementButtons = QtWidgets.QHBoxLayout()
        self.horizontalLayout_measurementButtons.addWidget(self.btn_loadMeasurement)
        self.horizontalLayout_measurementButtons.addWidget(self.btn_measureMeasurement)
        self.horizontalLayout_measurementButtons.addWidget(self.btn_calibrate)
        self.verticalLayout_main.addLayout(self.horizontalLayout_measurementButtons)
        self.verticalLayout_main.addWidget(self.listWidget_measurements)

        self.groupBox_saveOptions = QtWidgets.QGroupBox("Save Options", self)
        self.verticalLayout_saveOptions = QtWidgets.QVBoxLayout(self.groupBox_saveOptions)

        self.radioButton_saveRaw = QtWidgets.QRadioButton("Save Raw", self.groupBox_saveOptions)
        self.radioButton_saveCal = QtWidgets.QRadioButton("Save Cal", self.groupBox_saveOptions)
        self.radioButton_saveBoth = QtWidgets.QRadioButton("Save Both", self.groupBox_saveOptions)
        self.radioButton_saveBoth.setChecked(True)
        self.horizontalLayout_saveOptionsRadio = QtWidgets.QHBoxLayout()
        self.horizontalLayout_saveOptionsRadio.addWidget(self.radioButton_saveRaw)
        self.horizontalLayout_saveOptionsRadio.addWidget(self.radioButton_saveCal)
        self.horizontalLayout_saveOptionsRadio.addWidget(self.radioButton_saveBoth)
        self.verticalLayout_saveOptions.addLayout(self.horizontalLayout_saveOptionsRadio)

        self.btn_saveSelectedMeasurements = QtWidgets.QPushButton("Save Selected", self.groupBox_saveOptions)
        self.btn_saveAllMeasurements = QtWidgets.QPushButton("Save All", self.groupBox_saveOptions)
        self.horizontalLayout_saveMeasurementButtons = QtWidgets.QHBoxLayout()
        self.horizontalLayout_saveMeasurementButtons.addWidget(self.btn_saveSelectedMeasurements)
        self.horizontalLayout_saveMeasurementButtons.addWidget(self.btn_saveAllMeasurements)
        self.verticalLayout_saveOptions.addLayout(self.horizontalLayout_saveMeasurementButtons)

        self.verticalLayout_main.addWidget(self.groupBox_saveOptions)
        # --- END UI SETUP --- #

        self.btn_calibrate.clicked.connect(self.calibrate_measurements)
        self.btn_saveSelectedMeasurements.clicked.connect(self.listWidget_measurements.save_selected_items)
        self.btn_saveAllMeasurements.clicked.connect(self.listWidget_measurements.save_all_measurements)
        self.listWidget_measurements.get_save_which_mode = self.get_save_which_mode

    def connect_plot(self, ntwk_plot):
        self.listWidget_measurements.ntwk_plot = ntwk_plot

    def get_save_which_mode(self):
        if self.radioButton_saveRaw.isChecked():
            return "raw"
        elif self.radioButton_saveCal.isChecked():
            return "cal"
        else:
            return "both"

    def get_calibration(self):
        raise AttributeError("Must set get_calibration attribute to CalWidget.get_calibration method")

    def calibrate_measurements(self):
        calibration = self.get_calibration()
        for i in range(self.listWidget_measurements.count()):
            item = self.listWidget_measurements.item(i)  # type: widgets.NetworkListItem
            item.ntwk_corrected = calibration.apply_cal(item.ntwk)
            item.ntwk_corrected.name = item.ntwk.name + "-cal"
        self.listWidget_measurements.set_active_networks()


class TRLStandardsWidget(QtWidgets.QWidget):
    THRU_ID = "thru"
    SWITCH_TERMS_ID_FORWARD = "forward switch terms"
    SWITCH_TERMS_ID_REVERSE = "reverse switch terms"
    
    def __init__(self, parent=None):
        super(TRLStandardsWidget, self).__init__(parent)

        self.verticalLayout_main = QtWidgets.QVBoxLayout(self)
        self.verticalLayout_main.setContentsMargins(6, 6, 6, 6)

        self.label_thru = QtWidgets.QLabel("Thru")
        self.btn_measureThru = QtWidgets.QPushButton("Measure")
        self.btn_loadThru = QtWidgets.QPushButton("Load")
        self.horizontalLayout_thru = QtWidgets.QHBoxLayout()
        self.horizontalLayout_thru.addWidget(self.label_thru)
        self.horizontalLayout_thru.addWidget(self.btn_measureThru)
        self.horizontalLayout_thru.addWidget(self.btn_loadThru)
        self.verticalLayout_main.addLayout(self.horizontalLayout_thru)

        self.label_switchTerms = QtWidgets.QLabel("Switch Terms", self)
        self.btn_measureSwitchTerms = QtWidgets.QPushButton("Measure", self)
        self.btn_loadSwitchTerms = QtWidgets.QPushButton("Load", self)
        self.horizontalLayout_switchTerms = QtWidgets.QHBoxLayout()
        self.horizontalLayout_switchTerms.addWidget(self.label_switchTerms)
        self.horizontalLayout_switchTerms.addWidget(self.btn_measureSwitchTerms)
        self.horizontalLayout_switchTerms.addWidget(self.btn_loadSwitchTerms)
        self.verticalLayout_main.addLayout(self.horizontalLayout_switchTerms)

        self.listWidget_thru = widgets.NetworkListWidget(self)
        self.verticalLayout_main.addWidget(self.listWidget_thru)

        self.label_reflect = QtWidgets.QLabel("Reflect", self)
        self.btn_measureReflect = QtWidgets.QPushButton("Measure", self)
        self.btn_loadReflect = QtWidgets.QPushButton("Load", self)
        self.horizontalLayout_reflect = QtWidgets.QHBoxLayout()
        self.horizontalLayout_reflect.addWidget(self.label_reflect)
        self.horizontalLayout_reflect.addWidget(self.btn_measureReflect)
        self.horizontalLayout_reflect.addWidget(self.btn_loadReflect)
        self.verticalLayout_main.addLayout(self.horizontalLayout_reflect)

        self.listWidget_reflect = widgets.NetworkListWidget(self)
        self.verticalLayout_main.addWidget(self.listWidget_reflect)

        self.listWidget_line = widgets.NetworkListWidget(self)
        self.listWidget_line.MEASUREMENT_PREFIX = "line"
        self.label_line = QtWidgets.QLabel("Line")
        self.btn_measureLine = self.listWidget_line.get_measure_button()
        self.btn_loadLine = self.listWidget_line.get_load_button()
        self.horizontalLayout_line = QtWidgets.QHBoxLayout()
        self.horizontalLayout_line.addWidget(self.label_line)
        self.horizontalLayout_line.addWidget(self.btn_measureLine)
        self.horizontalLayout_line.addWidget(self.btn_loadLine)
        self.verticalLayout_main.addLayout(self.horizontalLayout_line)
        self.verticalLayout_main.addWidget(self.listWidget_line)

        self.btn_saveCalibration = QtWidgets.QPushButton("Save Cal", self)
        self.btn_loadCalibration = QtWidgets.QPushButton("Load Cal", self)
        self.horizontalLayout_saveCal = QtWidgets.QHBoxLayout()
        self.horizontalLayout_saveCal.addWidget(self.btn_saveCalibration)
        self.horizontalLayout_saveCal.addWidget(self.btn_loadCalibration)
        self.verticalLayout_main.addLayout(self.horizontalLayout_saveCal)
        # --- END UI SETUP --- #

        self.btn_loadThru.clicked.connect(self.load_thru)
        self.btn_loadReflect.clicked.connect(self.load_reflect)
        self.btn_loadSwitchTerms.clicked.connect(self.load_switch_terms)
        self.btn_measureThru.clicked.connect(self.measure_thru)
        self.btn_measureReflect.clicked.connect(self.measure_reflect)
        self.btn_measureSwitchTerms.clicked.connect(self.measure_switch_terms)
        self.btn_saveCalibration.clicked.connect(self.save_calibration)
        self.btn_loadCalibration.clicked.connect(self.load_calibration)

    def connect_plot(self, ntwk_plot):
        self.listWidget_thru.ntwk_plot = ntwk_plot
        self.listWidget_reflect.ntwk_plot = ntwk_plot
        self.listWidget_line.ntwk_plot = ntwk_plot

    def save_calibration(self):
        qt.warnMissingFeature()

    def load_calibration(self):
        qt.warnMissingFeature()

    def measure_thru(self):
        with self.vna_controller.get_analyzer() as nwa:
            ntwk = nwa.measure_twoport_ntwk()
        self.listWidget_thru.load_named_ntwk(ntwk, self.THRU_ID)

    def load_thru(self):
        self.listWidget_thru.load_named_ntwk(widgets.load_network_file(), self.THRU_ID)

    def measure_reflect(self):
        with self.vna_controller.get_analyzer() as nwa:
            self.load_reflect(nwa)

    def load_reflect(self, nwa=None):
        dialog = widgets.ReflectDialog(nwa)
        try:
            accepted = dialog.exec_()
            if accepted:
                if not dialog.reflect_2port.name:
                    # dialog.reflect_2port.name = self.listWidget_reflect.get_unique_name("reflect")
                    dialog.reflect_2port.name = "reflect"  # unique name will be assigned in load_network
                self.listWidget_reflect.load_network(dialog.reflect_2port)
        finally:
            dialog.close()

    def measure_switch_terms(self):
        with self.vna_controller.get_analyzer() as nwa:
            self.load_switch_terms(nwa)

    def load_switch_terms(self, nwa=None):
        dialog = widgets.SwitchTermsDialog(nwa)
        try:
            accepted = dialog.exec_()
            if accepted:
                self.listWidget_thru.load_named_ntwk(dialog.forward, self.SWITCH_TERMS_ID_FORWARD)
                self.listWidget_thru.load_named_ntwk(dialog.reverse, self.SWITCH_TERMS_ID_REVERSE)
        finally:
            dialog.close()

    def get_calibration(self):
        measured = []

        error_messages = []

        thru = self.listWidget_thru.get_named_item(self.THRU_ID).ntwk
        forward_switch_terms = self.listWidget_thru.get_named_item(self.SWITCH_TERMS_ID_FORWARD)
        reverse_switch_terms = self.listWidget_thru.get_named_item(self.SWITCH_TERMS_ID_REVERSE)
        if isinstance(forward_switch_terms, skrf.Network) and isinstance(reverse_switch_terms, skrf.Network):
            switch_terms = (forward_switch_terms.ntwk, reverse_switch_terms.ntwk)
        else:
            switch_terms = None
        
        if isinstance(thru, skrf.Network):
            measured.append(thru)
        else:
            error_messages.append("thru (type {:}) must be a valid 2-port network".format(type(thru)))

        reflects = self.listWidget_reflect.get_all_networks()
        if len(reflects) == 0:
            error_messages.append("missing reflect standards")
        else:
            measured.extend(reflects)
        
        lines = self.listWidget_line.get_all_networks()
        if len(lines) == 0:
            error_messages.append("missing line standards")
        else:
            measured.extend(lines)

        if len(error_messages) > 0:
            qt.error_popup("\n\n".join(error_messages))
            return

        return skrf.calibration.TRL(measured, n_reflects=len(reflects), switch_terms=switch_terms)
