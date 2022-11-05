from qtpy import QtWidgets, QtCore
import pyqtgraph as pg
import skrf
import numpy as np

from . import qt, networkListWidget, numeric_inputs, widgets, util


class CalibratedMeasurementsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.verticalLayout_main = QtWidgets.QVBoxLayout(self)
        self.verticalLayout_main.setContentsMargins(6, 6, 6, 6)

        self.listWidget_measurements = networkListWidget.NetworkListWidget()
        self.btn_measureMeasurement = self.listWidget_measurements.get_measure_button_twoport()
        self.btn_loadMeasurement = self.listWidget_measurements.get_load_button_twoport()
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
        super().__init__(parent)

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
        self.btn_measureLine = self.listWidget_line.get_measure_button_twoport()
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
            error_messages.append(f"thru (type {type(thru)}) must be a valid 2-port network")

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
        super().__init__(parent)
        self.setWindowTitle("Inspect Cal")
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.graphicsLayout = pg.GraphicsLayoutWidget()

        fHz = cal.frequency.f
        c = 299792458
        eps_eff = (cal.gamma * c / (1j * 2 * np.pi * fHz)) ** 2  # assuming non-magnetic TEM propagation
        Eratio_1cm = np.exp(-cal.gamma * 0.1)
        dbcm = 20 * np.log10(np.abs(Eratio_1cm))

        self.eps_plot = self.graphicsLayout.addPlot()  # type: pg.PlotItem
        self.eps_plot.setLabel("left", "effective permittivity")
        self.eps_plot.setLabel("bottom", "frequency", units="Hz")
        self.eps_plot.addLegend()
        self.eps_plot.showGrid(True, True)
        self.eps_plot.plot(fHz, eps_eff.real, pen=pg.mkPen("c"), name="real")
        self.eps_plot.plot(fHz, -eps_eff.imag, pen=pg.mkPen("y"), name="imag")

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
        super().__init__(parent)

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

        self.listWidget_thru = networkListWidget.NetworkListWidget()
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
        self.listWidget_reflect = networkListWidget.ParameterizedNetworkListWidget(item_parameters=refl_parameters)
        self.listWidget_reflect.label_parameters = ["refl_type", "offset"]
        self.verticalLayout_main.addWidget(self.listWidget_reflect)

        self.line_help = widgets.qt.HelpIndicator(title="Line Standards Help", help_text="""<h2>Line Standards</h2>
<p>You can have any number of line standards. &nbsp;The number of line standards is not entered, but instead is determined from the lines loaded or measured.</p>
<p>The accuracy of the calibration will in large part depend on having line standards that are not close to an integer multiple of 180 degrees out of phase from the calibration planes</p>
<h3>Parameters</h3>
<p>Lines have a length in mm. &nbsp;This can be edited by double clicking the items in the list below.</p>""")

        line_parameters = [{"name": "length", "type": "float", "default": 1.0, "units": "mm"}]
        self.listWidget_line = networkListWidget.ParameterizedNetworkListWidget(item_parameters=line_parameters)
        self.listWidget_line.label_parameters = ["length"]
        self.listWidget_line.name_prefix = "line"
        self.label_line = QtWidgets.QLabel("Line Standards")
        self.btn_measureLine = self.listWidget_line.get_measure_button_twoport()
        self.btn_loadLine = self.listWidget_line.get_load_button_twoport()
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
        self.btn_loadReflect.released.connect(self.load_reflect)
        self.btn_loadSwitchTerms.clicked.connect(self.load_switch_terms)
        self.btn_measureThru.released.connect(self.measure_thru)
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
        self.calibration_current = False
        self.vna_available = False

        self.btn_viewCalibration.setEnabled(False)
        self.btn_saveCalibration.setEnabled(False)
        self.btn_uploadCalibration.setEnabled(False)

        self.lineEdit_epsEstimate.value_changed.connect(self.calibration_parameters_changed)
        self.lineEdit_referencePlane.value_changed.connect(self.calibration_parameters_changed)
        self.lineEdit_thruLength.value_changed.connect(self.calibration_parameters_changed)
        self.comboBox_rootChoice.currentIndexChanged.connect(self.calibration_parameters_changed)
        self.listWidget_reflect.state_changed.connect(self.calibration_parameters_changed)
        self.listWidget_line.state_changed.connect(self.calibration_parameters_changed)
        self.listWidget_thru.state_changed.connect(self.calibration_parameters_changed)

    def calibration_parameters_changed(self):
        self.calibration_current = False
        self.btn_runCalibration.setEnabled(True)
        if self.calibration is not None:
            if self.vna_available:
                self.btn_uploadCalibration.setEnabled(True)
            self.btn_runCalibration.setText("Re-Run Cal")
            self.btn_viewCalibration.setText("View Cal (Old)")

    def view_calibration(self):
        dialog = NISTCalViewer(self.calibration)
        dialog.exec_()

    def upload_calibration(self):
        # TODO: enable specific parameters here, like which channel are we talking about?
        with self.get_analyzer() as vna:
            vna.upload_twoport_calibration(self.calibration, calname="skrf_NIST_mTRL")

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
        self.vna_available = available
        self.btn_measureLine.setEnabled(available)
        self.btn_measureSwitchTerms.setEnabled(available)
        self.btn_measureReflect.setEnabled(available)
        self.btn_measureThru.setEnabled(available)

    def connect_plot(self, ntwk_plot):
        self.listWidget_thru.ntwk_plot = ntwk_plot
        self.listWidget_reflect.ntwk_plot = ntwk_plot
        self.listWidget_line.ntwk_plot = ntwk_plot

    def save_calibration(self):
        """save an mTRL as a json + .s2p files in a zip archive"""
        if not self.calibration_current:
            msg = "Calibration parameters have changed and cal has not been re-run. " \
                  "This will save the previously run calibration.  Do you still want to save"
            reply = QtWidgets.QMessageBox.question(
                self, 'Cal Parameters Changed', msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.No:
                return

        cal_file = qt.getSaveFileName_Global("save calibration file", "skrf cal (*.zip)")
        if cal_file:
           self.calibration.save_calibration(cal_file)

    def load_calibration(self):
        cal_file = qt.getOpenFileName_Global("Open mTRL calibration File", "zip (*.zip)")
        if not cal_file:
            return

        cal = skrf.NISTMultilineTRL.load_calibration_archive(cal_file)
        self.calibration = cal

        self.listWidget_thru.load_named_ntwk(cal.measured[0], self.THRU_ID)
        if cal.switch_terms:
            self.listWidget_thru.load_named_ntwk(cal.switch_terms[0], self.SWITCH_TERMS_ID_FORWARD)
            self.listWidget_thru.load_named_ntwk(cal.switch_terms[1], self.SWITCH_TERMS_ID_REVERSE)

        n_reflects = len(cal.Grefls)
        reflects = cal.measured[1:n_reflects+1]
        refl_types = ["short" if rt == -1 else "open" for rt in cal.Grefls]
        refl_offsets = [roff * 1000 for roff in cal.refl_offset]
        self.listWidget_reflect.clear()
        for reflect, offset, rtype in zip(reflects, refl_offsets, refl_types):
            self.listWidget_reflect.load_network(reflect, False, parameters={"refl_type": rtype, "offset": offset})

        lines = cal.measured[1+n_reflects:]
        line_lengths = [l * 1000 for l in cal.l[1:]]
        self.listWidget_line.clear()
        for line, length in zip(lines, line_lengths):
            self.listWidget_line.load_network(line, False, parameters={"length": length})

        self.lineEdit_epsEstimate.set_value(cal.er_est)
        self.lineEdit_thruLength.set_value(cal.l[0] * 1000)
        self.lineEdit_referencePlane.set_value(cal.ref_plane * 1000)
        self.comboBox_rootChoice.setCurrentIndex(self.comboBox_rootChoice.findText(cal.gamma_root_choice))

    def measure_thru(self):
        with self.get_analyzer() as nwa:
            ntwk = nwa.get_twoport(**nwa.params_twoport)
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

    def set_calibration(self, cal):
        if not isinstance(cal, skrf.calibration.NISTMultilineTRL):
            raise TypeError("Must provide a NISTMultilineTRL Calibration object")
        else:
            self._calibration = cal
            self.calibration_current = True
            self.btn_runCalibration.setEnabled(False)
            self.btn_saveCalibration.setEnabled(True)
            self.btn_viewCalibration.setEnabled(True)
            self.btn_viewCalibration.setText("View Cal")
            self.btn_uploadCalibration.setEnabled(True)
            self.calibration_updated.emit(cal)

    calibration = property(get_calibration, set_calibration)

    def run_calibration_popup(self):
        dialog = qt.RunFunctionDialog(self.run_calibration, "Running Calibration",
                                      "Running NIST Multiline Cal\nthis window will close when finished")
        dialog.exec_()

    def run_calibration(self):
        self.btn_viewCalibration.setEnabled(False)

        measured = []

        error_messages = []

        thru = self.listWidget_thru.get_named_item(self.THRU_ID).ntwk

        try:
            forward_switch_terms = self.listWidget_thru.get_named_item(self.SWITCH_TERMS_ID_FORWARD).ntwk
            reverse_switch_terms = self.listWidget_thru.get_named_item(self.SWITCH_TERMS_ID_REVERSE).ntwk
        except Exception as e:
            forward_switch_terms = None
            reverse_switch_terms = None

        if isinstance(forward_switch_terms, skrf.Network) and isinstance(reverse_switch_terms, skrf.Network):
            switch_terms = (forward_switch_terms, reverse_switch_terms)
        else:
            switch_terms = None

        if isinstance(thru, skrf.Network):
            measured.append(thru)
        else:
            error_messages.append(f"thru (type {type(thru)}) must be a valid 2-port network")

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
        cal.run()

        self.calibration = cal
