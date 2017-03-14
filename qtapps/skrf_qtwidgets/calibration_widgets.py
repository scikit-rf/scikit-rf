from collections import OrderedDict
import json
import zipfile

from qtpy import QtWidgets, QtCore
import pyqtgraph as pg
import skrf
import numpy as np

from . import qt, networkListWidget, numeric_inputs, widgets, util


class CalibratedMeasurementsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(CalibratedMeasurementsWidget, self).__init__(parent)

        self.verticalLayout_main = QtWidgets.QVBoxLayout(self)
        self.verticalLayout_main.setContentsMargins(6, 6, 6, 6)

        self.listWidget_measurements = networkListWidget.NetworkListWidget(self)
        self.btn_measureMeasurement = self.listWidget_measurements.get_measure_button()
        self.btn_loadMeasurement = self.listWidget_measurements.get_load_button()
        self.horizontalLayout_measurementButtons = QtWidgets.QHBoxLayout()
        self.horizontalLayout_measurementButtons.addWidget(self.btn_loadMeasurement)
        self.horizontalLayout_measurementButtons.addWidget(self.btn_measureMeasurement)
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

        self.btn_saveSelectedMeasurements.clicked.connect(self.listWidget_measurements.save_selected_items)
        self.btn_saveAllMeasurements.clicked.connect(self.listWidget_measurements.save_all_measurements)
        self.listWidget_measurements.get_save_which_mode = self.get_save_which_mode
        self.listWidget_measurements.correction = self.correction
        self._get_analyzer = None
        self._calibration = None

    def connect_plot(self, ntwk_plot):
        self.listWidget_measurements.ntwk_plot = ntwk_plot

    @property
    def get_analyzer(self):
        if self._get_analyzer is None:
            raise AttributeError("get_analyzer method not set")
        return self._get_analyzer

    @get_analyzer.setter
    def get_analyzer(self, get_analyzer_method):
        """
        set the method used to get an instance of a network analyzer driver

        Parameters
        ----------
        get_analyzer_method : object method
            the method used to return an skrf.vi.vna.VNA driver object, typically retrieved from a VNAController object
        """
        self._get_analyzer = get_analyzer_method
        self.listWidget_measurements.get_analyzer = get_analyzer_method

    def get_save_which_mode(self):
        if self.radioButton_saveRaw.isChecked():
            return "raw"
        elif self.radioButton_saveCal.isChecked():
            return "cal"
        else:
            return "both"

    def get_calibration(self):
        return self._calibration

    def set_calibration(self, calibration):
        if isinstance(calibration, skrf.Calibration):
            self._calibration = calibration
            self.calibrate_measurements()
        else:
            raise TypeError("must provide a valid skrf.Calibration object")

    calibration = property(get_calibration, set_calibration)

    def correction(self, ntwk):
        if self.calibration is None:
            return None
        else:
            return self.calibration.apply_cal(ntwk)

    def calibrate_measurements(self):
        if self.calibration is None:
            return

        for i in range(self.listWidget_measurements.count()):
            item = self.listWidget_measurements.item(i)  # type: networkListWidget.NetworkListItem
            item.ntwk_corrected = self.calibration.apply_cal(item.ntwk)
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

        self.listWidget_thru = networkListWidget.NetworkListWidget(self)
        self.verticalLayout_main.addWidget(self.listWidget_thru)

        self.label_reflect = QtWidgets.QLabel("Reflect", self)
        self.btn_measureReflect = QtWidgets.QPushButton("Measure", self)
        self.btn_loadReflect = QtWidgets.QPushButton("Load", self)
        self.horizontalLayout_reflect = QtWidgets.QHBoxLayout()
        self.horizontalLayout_reflect.addWidget(self.label_reflect)
        self.horizontalLayout_reflect.addWidget(self.btn_measureReflect)
        self.horizontalLayout_reflect.addWidget(self.btn_loadReflect)
        self.verticalLayout_main.addLayout(self.horizontalLayout_reflect)

        self.listWidget_reflect = networkListWidget.NetworkListWidget(self)
        self.verticalLayout_main.addWidget(self.listWidget_reflect)

        self.listWidget_line = networkListWidget.NetworkListWidget(self)
        self.listWidget_line.name_prefix = "line"
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
        self._get_analyzer = None

    @property
    def get_analyzer(self):
        if self._get_analyzer is None:
            raise AttributeError("get_analyzer method not set")
        return self._get_analyzer

    @get_analyzer.setter
    def get_analyzer(self, get_analyzer_method):
        """
        set the method used to get an instance of a network analyzer driver

        Parameters
        ----------
        get_analyzer_method : object method
            the method used to return an skrf.vi.vna.VNA driver object, typically retrieved from a VNAController object
        """
        self._get_analyzer = get_analyzer_method
        self.listWidget_reflect.get_analyzer = get_analyzer_method
        self.listWidget_line.get_analyzer = get_analyzer_method
        self.listWidget_thru.get_analyzer = get_analyzer_method

    def connect_plot(self, ntwk_plot):
        self.listWidget_thru.ntwk_plot = ntwk_plot
        self.listWidget_reflect.ntwk_plot = ntwk_plot
        self.listWidget_line.ntwk_plot = ntwk_plot

    def save_calibration(self):
        qt.warnMissingFeature()

    def load_calibration(self):
        qt.warnMissingFeature()

    def measure_thru(self):
        with self.get_analyzer() as nwa:
            ntwk = nwa.measure_twoport_ntwk()
        self.listWidget_thru.load_named_ntwk(ntwk, self.THRU_ID)

    def load_thru(self):
        thru = widgets.load_network_file()
        if isinstance(thru, skrf.Network):
            self.listWidget_thru.load_named_ntwk(thru, self.THRU_ID)

    def measure_reflect(self):
        with self.get_analyzer() as nwa:
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
        with self.get_analyzer() as nwa:
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


class NISTCalViewer(QtWidgets.QDialog):
    def __init__(self, cal, parent=None):
        """
        a simple viewer to check the quality of the calibration

        Parameters
        ----------
        cal : skrf.calibration.NISTMultilineTRL
            the calibration object
        """
        super(NISTCalViewer, self).__init__(parent)
        self.setWindowTitle("Inspect Cal")

        fHz = cal.frequency.f
        c = 299792458

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.graphicsLayout = pg.GraphicsLayoutWidget()

        eps_eff = (cal.gamma * c / (1j * 2 * np.pi * fHz)) ** 2  # assuming non-magnetic TEM propagation
        self.eps_plot = self.graphicsLayout.addPlot()  # type: pg.PlotItem
        self.eps_plot.setLabel("left", "effective permittivity")
        self.eps_plot.setLabel("bottom", "frequency", units="Hz")
        self.eps_plot.addLegend()
        self.eps_plot.showGrid(True, True)
        self.eps_plot.plot(fHz, eps_eff.real, pen=pg.mkPen("c"), name="real")
        self.eps_plot.plot(fHz, -eps_eff.imag, pen=pg.mkPen("y"), name="imag")

        Eratio_1cm = np.exp(-cal.gamma * 0.1)
        dbcm = 20 * np.log10(np.abs(Eratio_1cm))
        self.y_plot = self.graphicsLayout.addPlot()  # type: pg.PlotItem
        self.y_plot.setLabel("left", "line loss", units="dB/cm")
        self.y_plot.setLabel("bottom", "frequency", units="Hz")
        self.y_plot.showGrid(True, True)
        self.y_plot.plot(fHz, dbcm)

        self.layout.addWidget(self.graphicsLayout)

        self.resize(900, 400)


class NISTTRLStandardsWidget(QtWidgets.QWidget):
    THRU_ID = "thru"
    SWITCH_TERMS_ID_FORWARD = "forward switch terms"
    SWITCH_TERMS_ID_REVERSE = "reverse switch terms"

    calibration_updated = QtCore.Signal(object)

    def __init__(self, parent=None):
        super(NISTTRLStandardsWidget, self).__init__(parent)

        self.verticalLayout_main = QtWidgets.QVBoxLayout(self)
        self.verticalLayout_main.setContentsMargins(6, 6, 6, 6)

        self.lineEdit_epsEstimate = numeric_inputs.DoubleLineEdit(1.0)
        self.comboBox_rootChoice = QtWidgets.QComboBox()
        self.comboBox_rootChoice.addItems(("real", "imag", "auto"))
        self.lineEdit_thruLength = numeric_inputs.InputWithUnits("mm", 0)
        self.lineEdit_referencePlane = numeric_inputs.InputWithUnits("mm", 0)

        self.label_epsEstimate = QtWidgets.QLabel("eps est.")
        self.label_rootChoice = QtWidgets.QLabel("root choice")
        self.label_thruLength = QtWidgets.QLabel("thru length (mm)")
        self.label_referencePlane = QtWidgets.QLabel("ref plane shift (mm)")
        self.groupBox_calOptions = QtWidgets.QGroupBox("Calibration Options")

        col1 = self.verticalLayout_parametersCol1 = QtWidgets.QVBoxLayout()
        col2 = self.verticalLayout_parametersCol2 = QtWidgets.QVBoxLayout()
        col3 = self.verticalLayout_parametersCol3 = QtWidgets.QVBoxLayout()
        col4 = self.verticalLayout_parametersCol4 = QtWidgets.QVBoxLayout()
        col1.addWidget(self.label_thruLength)
        col2.addWidget(self.lineEdit_thruLength)
        col1.addWidget(self.label_referencePlane)
        col2.addWidget(self.lineEdit_referencePlane)
        col3.addWidget(self.label_epsEstimate)
        col4.addWidget(self.lineEdit_epsEstimate)
        col3.addWidget(self.label_rootChoice)
        col4.addWidget(self.comboBox_rootChoice)

        self.horizontalLayout_parameters = QtWidgets.QHBoxLayout(self.groupBox_calOptions)
        self.horizontalLayout_parameters.addLayout(self.verticalLayout_parametersCol1)
        self.horizontalLayout_parameters.addLayout(self.verticalLayout_parametersCol2)
        self.horizontalLayout_parameters.addLayout(self.verticalLayout_parametersCol3)
        self.horizontalLayout_parameters.addLayout(self.verticalLayout_parametersCol4)
        self.verticalLayout_main.addWidget(self.groupBox_calOptions)

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

        self.listWidget_thru = networkListWidget.NetworkListWidget(self)
        self.verticalLayout_main.addWidget(self.listWidget_thru)

        self.reflect_help = widgets.qt.HelpIndicator(title="Reflect Standards Help", help_text="""<h2>Reflect Standards</h2>
            <p>You can have any number of reflect standards. &nbsp;The number of standards is not entered,
            but rather is determined from the number that you load or measure</p>
            <h3>Parameters</h3>
            <p>Reflect standards can have 2 parameters:
            <ul><li>offset: specified in mm</li><li>type: open/short.</li></ul>
            You can edit these parameters by double clicking on the items in the list.</p>""")
        self.label_reflect = QtWidgets.QLabel("Reflect Standards", self)
        self.btn_measureReflect = QtWidgets.QPushButton("Measure", self)
        self.btn_loadReflect = QtWidgets.QPushButton("Load", self)
        self.horizontalLayout_reflect = QtWidgets.QHBoxLayout()
        self.horizontalLayout_reflect.addWidget(self.reflect_help)
        self.horizontalLayout_reflect.addWidget(self.label_reflect)
        self.horizontalLayout_reflect.addWidget(self.btn_measureReflect)
        self.horizontalLayout_reflect.addWidget(self.btn_loadReflect)
        self.verticalLayout_main.addLayout(self.horizontalLayout_reflect)

        refl_parameters = [
            {"name": "refl_type", "type": "str", "default": "short", "combo_list": ["short", "open"]},
            {"name": "offset", "type": "float", "default": 0.0, "units": "mm"}
        ]
        self.listWidget_reflect = networkListWidget.ParameterizedNetworkListWidget(self, refl_parameters)
        self.listWidget_reflect.label_parameters = ["refl_type", "offset"]
        self.verticalLayout_main.addWidget(self.listWidget_reflect)

        self.line_help = widgets.qt.HelpIndicator(title="Line Standards Help", help_text="""<h2>Line Standards</h2>
<p>You can have any number of line standards. &nbsp;The number of line standards is not entered, but instead is determined from the lines loaded or measured.</p>
<p>The accuracy of the calibration will in large part depend on having line standards that are not close to an integer multiple of 180 degrees out of phase from the calibration planes</p>
<h3>Parameters</h3>
<p>Lines have a length in mm. &nbsp;This can be edited by double clicking the items in the list below.</p>""")

        line_parameters = [{"name": "length", "type": "float", "default": 1.0, "units": "mm"}]
        self.listWidget_line = networkListWidget.ParameterizedNetworkListWidget(self, line_parameters)
        self.listWidget_line.label_parameters = ["length"]
        self.listWidget_line.name_prefix = "line"
        self.label_line = QtWidgets.QLabel("Line Standards")
        self.btn_measureLine = self.listWidget_line.get_measure_button()
        self.btn_loadLine = self.listWidget_line.get_load_button()
        self.horizontalLayout_line = QtWidgets.QHBoxLayout()
        self.horizontalLayout_line.addWidget(self.line_help)
        self.horizontalLayout_line.addWidget(self.label_line)
        self.horizontalLayout_line.addWidget(self.btn_measureLine)
        self.horizontalLayout_line.addWidget(self.btn_loadLine)
        self.verticalLayout_main.addLayout(self.horizontalLayout_line)
        self.verticalLayout_main.addWidget(self.listWidget_line)

        self.btn_saveCalibration = QtWidgets.QPushButton("Save Cal File")
        self.btn_loadCalibration = QtWidgets.QPushButton("Load Cal File")
        self.horizontalLayout_cal1 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_cal1.addWidget(self.btn_saveCalibration)
        self.horizontalLayout_cal1.addWidget(self.btn_loadCalibration)
        self.verticalLayout_main.addLayout(self.horizontalLayout_cal1)

        self.btn_runCalibration = QtWidgets.QPushButton("Run Cal")
        self.btn_viewCalibration = QtWidgets.QPushButton("View Cal")
        self.btn_uploadCalibration = QtWidgets.QPushButton("Upload Cal")
        self.horizontalLayout_cal2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_cal2.addWidget(self.btn_runCalibration)
        self.horizontalLayout_cal2.addWidget(self.btn_viewCalibration)
        self.horizontalLayout_cal2.addWidget(self.btn_uploadCalibration)
        self.verticalLayout_main.addLayout(self.horizontalLayout_cal2)
        # --- END UI SETUP --- #

        self.btn_loadThru.clicked.connect(self.load_thru)
        self.btn_loadReflect.clicked.connect(self.load_reflect)
        self.btn_loadSwitchTerms.clicked.connect(self.load_switch_terms)
        self.btn_measureThru.clicked.connect(self.measure_thru)
        self.btn_measureReflect.clicked.connect(self.measure_reflect)
        self.btn_measureSwitchTerms.clicked.connect(self.measure_switch_terms)
        self.btn_saveCalibration.clicked.connect(self.save_calibration)
        self.btn_loadCalibration.clicked.connect(self.load_calibration)
        # self.btn_runCalibration.clicked.connect(self.run_calibration_popup)
        self.btn_runCalibration.clicked.connect(self.run_calibration)
        self.btn_viewCalibration.clicked.connect(self.view_calibration)
        self.btn_uploadCalibration.clicked.connect(self.upload_calibration)
        self._get_analyzer = None
        self._calibration = None

        self.btn_viewCalibration.setEnabled(False)

    def view_calibration(self):
        dialog = NISTCalViewer(self.get_calibration())
        dialog.exec_()

    def upload_calibration(self):
        qt.warnMissingFeature()

    @property
    def get_analyzer(self):
        if self._get_analyzer is None:
            raise AttributeError("get_analyzer method not set")
        return self._get_analyzer

    @get_analyzer.setter
    def get_analyzer(self, get_analyzer_method):
        """
        set the method used to get an instance of a network analyzer driver

        Parameters
        ----------
        get_analyzer_method : object method
            the method used to return an skrf.vi.vna.VNA driver object, typically retrieved from a VNAController object
        """
        self._get_analyzer = get_analyzer_method
        self.listWidget_reflect.get_analyzer = get_analyzer_method
        self.listWidget_line.get_analyzer = get_analyzer_method
        self.listWidget_thru.get_analyzer = get_analyzer_method

    def process_vna_available(self, available):
        """
        Parameters
        ----------
        available : bool
        """
        self.btn_measureLine.setEnabled(available)
        self.btn_measureSwitchTerms.setEnabled(available)
        self.btn_measureReflect.setEnabled(available)
        self.btn_measureThru.setEnabled(available)
        self.btn_uploadCalibration.setEnabled(available)

    def connect_plot(self, ntwk_plot):
        self.listWidget_thru.ntwk_plot = ntwk_plot
        self.listWidget_reflect.ntwk_plot = ntwk_plot
        self.listWidget_line.ntwk_plot = ntwk_plot

    def save_calibration(self):
        """save an mTRL as a json + .s2p files in a zip archive"""
        thru = self.listWidget_thru.get_named_item(self.THRU_ID).ntwk
        fswitch = self.listWidget_thru.get_named_item(self.SWITCH_TERMS_ID_FORWARD).ntwk
        rswitch = self.listWidget_thru.get_named_item(self.SWITCH_TERMS_ID_REVERSE).ntwk
        reflects = self.listWidget_reflect.get_all_networks()
        reflect_offsets = self.listWidget_reflect.get_parameter_from_all("offset")
        reflect_types = self.listWidget_reflect.get_parameter_from_all("refl_type")
        lines = self.listWidget_line.get_all_networks()
        line_lengths = self.listWidget_line.get_parameter_from_all("length")

        switch_terms = True if isinstance(fswitch, skrf.Network) and isinstance(fswitch, skrf.Network) else False

        # --- make sure we have no naming conflicts --- #
        ntwk_names = [thru.name]
        if not switch_terms:
            ntwk_names.extend(("", ""))
        else:
            ntwk_names.extend((fswitch.name, rswitch.name))

        ntwk_names.extend(reflect.name for reflect in reflects)
        ntwk_names.extend(line.name for line in lines)

        n_reflects = len(reflects)
        n_lines = len(lines)

        rx = 3  # starting index of the reflects
        lx = rx + n_reflects  # starting index of the lines

        for i, name in enumerate(ntwk_names[rx:]):
            ntwk_names[rx+i] = util.unique_name(name, ntwk_names, exclude=rx+i)
        # --- done with naming conflicts --- #

        cal_parameters = OrderedDict()
        cal_standards = OrderedDict()
        cal_standards["thru"] = {"name": ntwk_names[0], "length": self.lineEdit_thruLength.get_value(units="m")}
        if switch_terms:
            cal_standards["forward switch terms"] = {"name": ntwk_names[1]}
            cal_standards["reverse switch terms"] = {"name": ntwk_names[2]}
        else:
            cal_standards["forward switch terms"] = None
            cal_standards["reverse switch terms"] = None

        cal_standards["reflects"] = [None] * n_reflects
        for i, name in enumerate(ntwk_names[rx:rx+n_reflects]):
            cal_standards["reflects"][i] = {
                "name": name,
                "refl_type": reflect_types[i],
                "offset": reflect_offsets[i] / 1000
            }

        cal_standards["lines"] = [None] * n_lines
        for i, name in enumerate(ntwk_names[lx:]):
            cal_standards["lines"][i] = {
                "name": name,
                "length": line_lengths[i] / 1000
            }

        cal_parameters["parameters"] = {
            "reference plane shift": self.lineEdit_referencePlane.get_value(units="m"),
            "eps estimate": self.lineEdit_epsEstimate.get_value(),
            "root choice": self.comboBox_rootChoice.currentText(),
            "port1 length estimate": 0,
            "port2 length estimate": 0
        }
        cal_parameters["standards"] = cal_standards
        l = [cal_standards["thru"]["length"]]
        l.extend(l / 1000 for l in line_lengths)
        cal_parameters["skrf kwargs"] = {
            "Grefls": [-1 if rtype == "short" else 1 for rtype in reflect_types],
            "l": l,
            "er_est": cal_parameters["parameters"]["eps estimate"],
            "refl_offset": [r / 1000 for r in reflect_offsets],
            "p1_len_est": 0, "p2_len_est": 0,
            "ref_plane": cal_parameters["parameters"]["reference plane shift"],
            "gamma_root_choice": cal_parameters["parameters"]["root choice"]
        }

        cal_file = qt.getSaveFileName_Global("save calibration file", "skrf cal (*.zip)")
        if not cal_file:
            return

        with zipfile.ZipFile(cal_file, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("parameters.json", json.dumps(cal_parameters, indent=2))
            thru_str = util.snp_string(thru, comments="Thru Standard")
            fswitch_str = util.snp_string(fswitch, comments="Forward Switch Terms")
            rswitch_str = util.snp_string(rswitch, comments="Reverse Switch Terms")
            archive.writestr(ntwk_names[0] + ".s2p", thru_str)
            archive.writestr(ntwk_names[1] + ".s1p", fswitch_str)
            archive.writestr(ntwk_names[2] + ".s1p", rswitch_str)
            for i, name in enumerate(ntwk_names[rx:rx+n_reflects]):
                reflect_str = util.snp_string(
                    reflects[i], comments=["offset={} mm".format(reflect_offsets[i]), "type={}".format(reflect_types)])
                archive.writestr(name + ".s2p", reflect_str)
            for i, name in enumerate(ntwk_names[lx:]):
                line_str = util.snp_string(lines[i], comments="length={} mm".format(line_lengths[i]))
                archive.writestr(name + ".s2p", line_str)

    def load_calibration(self):
        cal_file = qt.getOpenFileName_Global("Open mTRL calibration File", "zip (*.zip)")
        if not cal_file:
            return

        with zipfile.ZipFile(cal_file) as archive:
            params = json.loads(archive.open('parameters.json').read().decode("ascii"))
            networks = util.read_zipped_touchstones(archive)

        standards = params["standards"]
        thru = networks[standards["thru"]["name"]]
        fswitch = networks[standards["forward switch terms"]["name"]]
        rswitch = networks[standards["reverse switch terms"]["name"]]

        self.listWidget_thru.load_named_ntwk(thru, thru.name)
        self.listWidget_thru.load_named_ntwk(fswitch, fswitch.name)
        self.listWidget_thru.load_named_ntwk(rswitch, rswitch.name)

        self.listWidget_reflect.clear()
        for reflect in standards["reflects"]:
            ntwk = networks[reflect["name"]]
            offset = reflect["offset"] * 1000
            rtype = reflect["refl_type"]
            self.listWidget_reflect.load_network(ntwk, False, parameters={"refl_type": rtype, "offset": offset})

        self.listWidget_line.clear()
        for line in standards["lines"]:
            ntwk = networks[line["name"]]
            length = line["length"] * 1000
            self.listWidget_line.load_network(ntwk, False, parameters={"length": length})

        self.lineEdit_epsEstimate.set_value(params["parameters"]["eps estimate"])
        self.lineEdit_thruLength.set_value(standards["thru"]["length"] * 1000)
        self.lineEdit_referencePlane.set_value(params["parameters"]["reference plane shift"] * 1000)
        self.comboBox_rootChoice.setCurrentIndex(self.comboBox_rootChoice.findText(params["parameters"]["root choice"]))

    def measure_thru(self):
        with self.get_analyzer() as nwa:
            ntwk = nwa.get_twoport()
        self.listWidget_thru.load_named_ntwk(ntwk, self.THRU_ID)

    def load_thru(self):
        thru = widgets.load_network_file()
        if isinstance(thru, skrf.Network):
            self.listWidget_thru.load_named_ntwk(thru, self.THRU_ID)

    def measure_reflect(self):
        with self.get_analyzer() as nwa:
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
        with self.get_analyzer() as nwa:
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
        return self._calibration

    def run_calibration_popup(self):
        dialog = qt.RunFunctionDialog(self.run_calibration, "Running Calibration",
                                      "Running NIST Multiline Cal\nthis window will close when finished")
        dialog.exec_()

    def run_calibration(self):
        self.btn_viewCalibration.setEnabled(False)

        measured = []

        error_messages = []

        thru = self.listWidget_thru.get_named_item(self.THRU_ID).ntwk
        forward_switch_terms = self.listWidget_thru.get_named_item(self.SWITCH_TERMS_ID_FORWARD).ntwk
        reverse_switch_terms = self.listWidget_thru.get_named_item(self.SWITCH_TERMS_ID_REVERSE).ntwk
        if isinstance(forward_switch_terms, skrf.Network) and isinstance(reverse_switch_terms, skrf.Network):
            switch_terms = (forward_switch_terms, reverse_switch_terms)
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

        refl_types = self.listWidget_reflect.get_parameter_from_all("refl_type")
        Grefls = [-1 if rtype == "short" else 1 for rtype in refl_types]
        offsets = self.listWidget_reflect.get_parameter_from_all("offset")
        refl_offset = [roff / 1000 for roff in offsets]  # convert mm to m

        line_lengths = self.listWidget_line.get_parameter_from_all("length")
        l = [length / 1000 for length in line_lengths]  # convert mm to m
        l.insert(0, self.lineEdit_thruLength.get_value("m"))

        er_est = self.lineEdit_epsEstimate.get_value()
        ref_plane = self.lineEdit_referencePlane.get_value("m")
        gamma_root_choice = self.comboBox_rootChoice.currentText()

        cal = skrf.calibration.NISTMultilineTRL(
            measured, Grefls, l,
            er_est=er_est, refl_offset=refl_offset, # p1_len_est=p1_len_est, p2_len_est=p2_len_est,
            ref_plane=ref_plane, gamma_root_choice=gamma_root_choice, switch_terms=switch_terms
        )

        self._calibration = cal
        self.btn_viewCalibration.setEnabled(True)
        self.calibration_updated.emit(cal)
