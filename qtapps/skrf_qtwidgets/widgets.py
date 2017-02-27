import os
import sys
import traceback

from qtpy import QtWidgets, QtCore, QtGui

import skrf
from . import numeric_inputs
from . import qt
from .analyzers import analyzers


def load_network_file(caption="load network file", filter="touchstone file (*.s*p)"):
    fname = qt.getOpenFileName_Global(caption, filter)
    if not fname:
        return None

    try:
        ntwk = skrf.Network(fname)
    except Exception as e:
        qt.error_popup(e)
        return None

    return ntwk


def load_network_files(caption="load network file", filter="touchstone file (*.s*p)"):
    fnames = qt.getOpenFileNames_Global(caption, filter)
    if not fnames:
        return None

    ntwks = []
    errors = []

    for fname in fnames:
        try:
            ntwks.append(skrf.Network(fname))
        except Exception:
            etype, value, tb = sys.exc_info()
            errors.append(fname + ": " + traceback.format_exception_only(etype, value))

    if errors:
        qt.error_popup(errors)

    return ntwks


def save_multiple_networks(ntwk_list):
    dirname = qt.getDirName_Global("select directory to save network files")
    if not dirname:
        return

    remember = False
    overwrite = False
    for ntwk in ntwk_list:
        if isinstance(ntwk, skrf.Network):
            fname = os.path.join(dirname, ntwk.name) + ".s{:d}p".format(ntwk.s.shape[1])
            if os.path.isfile(fname):
                if not remember:
                    msg = "The file:\n" + fname + "\nalready exists.\n\nDo you want to overwrite the file?"
                    dialog = OverwriteFilesQuery(title="File Already Exists", msg=msg)
                    dialog.exec_()
                    if dialog.choice == "yes":
                        overwrite = True
                    elif dialog.choice == "yes to all":
                        overwrite = True
                        remember = True
                    elif dialog.choice == "no":
                        overwrite = False
                    elif dialog.choice == "cancel":
                        return
                    else:
                        raise ValueError("did not recognize dialog choice")

                if not overwrite:
                    filter = "Touchstone file (*.s{:d}p)".format(ntwk.s.shape[1])
                    fname = qt.getSaveFileName_Global("save network file", filter)

            ntwk.write_touchstone(fname)


class OverwriteFilesQuery(QtWidgets.QDialog):
    def __init__(self, title="", msg="", parent=None):
        super(OverwriteFilesQuery, self).__init__(parent)
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.textBrowser = QtWidgets.QTextBrowser(self)
        self.verticalLayout.addWidget(self.textBrowser)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.No|QtWidgets.QDialogButtonBox.Yes|QtWidgets.QDialogButtonBox.YesToAll)

        self.choice = None

        self.yes = self.buttonBox.button(QtWidgets.QDialogButtonBox.Yes)
        self.yesToAll = self.buttonBox.button(QtWidgets.QDialogButtonBox.YesToAll)
        self.no = self.buttonBox.button(QtWidgets.QDialogButtonBox.No)
        self.cancel = self.buttonBox.button(QtWidgets.QDialogButtonBox.Cancel)
        self.yes.clicked.connect(self.set_yes)
        self.yesToAll.clicked.connect(self.set_yesToAll)
        self.no.clicked.connect(self.set_no)
        self.cancel.clicked.connect(self.set_cancel)

        self.horizontalLayout.addWidget(self.buttonBox)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.setWindowTitle(title)
        self.textBrowser.setText(msg)

    def set_yes(self):
        self.choice = "yes"

    def set_yesToAll(self):
        self.choice = "yes to all"

    def set_no(self):
        self.choice = "no"

    def set_cancel(self):
        self.choice = "cancel"


class NetworkParameterEditor(QtWidgets.QDialog):
    def __init__(self, item, item_parameters, window_title="Edit Parameters", parent=None):
        super(NetworkParameterEditor, self).__init__(parent)
        self.setWindowTitle(window_title)
        vlay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout(None)
        vlay.addLayout(form)

        self.inputs = dict()

        input = QtWidgets.QLineEdit()
        input.setText(item.ntwk.name)
        form.addRow("name", input)
        self.inputs["name"] = input

        for name, param in item_parameters.items():
            value = item.parameters[name]
            if param["combo_list"]:
                input = QtWidgets.QComboBox()
                input.addItems(param["combo_list"])
                input.setCurrentIndex(input.findText(value))
                row_name = name
            elif param["units"] and param["type"] in ("int", "float"):
                input = numeric_inputs.InputWithUnits(param["units"])
                input.setText(str(value))
                row_name = "{:} ({:})".format(name, param["units"])
            else:
                input = QtWidgets.QLineEdit()
                input.setText(str(value))
                row_name = name

            self.inputs[name] = input
            form.addRow(row_name, input)

        ok = QtWidgets.QPushButton("Ok")
        ok.setAutoDefault(False)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.setAutoDefault(False)
        hlay = QtWidgets.QHBoxLayout()
        hlay.addWidget(ok)
        hlay.addWidget(cancel)
        vlay.addLayout(hlay)
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)

    def showEvent(self, event):
        self.resize(self.width() * 1.25, self.height())
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class SwitchTermsDialog(QtWidgets.QDialog):
    def __init__(self, analyzer=None, parent=None):
        super(SwitchTermsDialog, self).__init__(parent)

        self.setWindowTitle("Measure Switch Terms")

        self.verticalLayout = QtWidgets.QVBoxLayout(self)

        self.btn_measureSwitch = QtWidgets.QPushButton("Measure Switch Terms")
        self.label_measureSwitch = QtWidgets.QLabel("Not Measured")
        self.btn_loadForwardSwitch = QtWidgets.QPushButton("Load Forward Switch Terms")
        self.label_loadForwardSwitch = QtWidgets.QLabel("Not Measured")
        self.btn_loadReverseSwitch = QtWidgets.QPushButton("Load Reverse Switch Terms")
        self.label_loadReverseSwitch = QtWidgets.QLabel("Not Measured")

        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.addWidget(self.btn_measureSwitch, 0, 0)
        self.gridLayout.addWidget(self.label_measureSwitch, 0, 1)
        self.gridLayout.addWidget(self.btn_loadForwardSwitch, 1, 0)
        self.gridLayout.addWidget(self.label_loadForwardSwitch, 1, 1)
        self.gridLayout.addWidget(self.btn_loadReverseSwitch, 2, 0)
        self.gridLayout.addWidget(self.label_loadReverseSwitch, 2, 1)

        self.verticalLayout.addLayout(self.gridLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.verticalLayout.addWidget(self.buttonBox)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.analyzer = analyzer
        if self.analyzer is None:
            self.btn_measureSwitch.setEnabled(False)
        self.forward = None
        self.reverse = None
        self._ready = False
        self.current_item = None

        self.btn_measureSwitch.clicked.connect(self.measure_switch)
        self.btn_loadForwardSwitch.clicked.connect(self.load_forward_switch)
        self.btn_loadReverseSwitch.clicked.connect(self.load_reverse_switch)

        self.ok = self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok)  # type: QtWidgets.QPushButton
        self.ok.setEnabled(False)

    def measure_switch(self):
        self.forward, self.reverse = self.analyzer.get_switch_terms()
        self.evaluate()

    def load_forward_switch(self):
        self.forward = load_network_file("Load Forward Switch Terms", "Touchstone 1-port (*.s1p)")
        if type(self.forward) is not skrf.Network:
            self.forward = None

        self.evaluate()

    def load_reverse_switch(self):
        self.reverse = load_network_file("Load Reverse Switch Terms", "Touchstone 1-port (*.s1p)")
        if type(self.reverse) is not skrf.Network:
            self.reverse = None

        self.evaluate()

    @property
    def ready(self): return self._ready

    @ready.setter
    def ready(self, val):
        if val is True:
            self._ready = True
            self.ok.setEnabled(True)
        else:
            self._ready = False
            self.ok.setEnabled(False)

    def evaluate(self):
        if type(self.forward) is skrf.Network:
            self.label_loadForwardSwitch.setText("forward - measured")
        else:
            self.label_loadForwardSwitch.setText("forward - not measured")

        if type(self.reverse) is skrf.Network:
            self.label_loadReverseSwitch.setText("reverse - measured")
        else:
            self.label_loadReverseSwitch.setText("reverse - not measured")

        if type(self.forward) is skrf.Network and type(self.reverse) is skrf.Network:
            self.label_measureSwitch.setText("measured")
            self.ready = True


class VnaController(QtWidgets.QWidget):
    FUNITS = ["Hz", "kHz", "MHz", "GHz", "THz", "PHz"]
    FCONVERSIONS = {"Hz": 1., "kHz": 1e-3, "MHz": 1e-6, "GHz": 1e-9, "THz": 1e-12, "PHz": 1e-15}

    def __init__(self, parent=None):
        super(VnaController, self).__init__(parent)

        # --- Setup UI Elements --- #
        self.verticalLayout = QtWidgets.QVBoxLayout(self)  # primary widget layout
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)  # normally this will be embedded in another application

        self.checkBox_TriggerNew = QtWidgets.QCheckBox("Trigger New", self)

        self.label_analyzerList = QtWidgets.QLabel("Select Analyzer", self)
        self.comboBox_analyzer = QtWidgets.QComboBox(self)
        self.hlay_analyzerList = QtWidgets.QHBoxLayout()
        self.hlay_analyzerList.addWidget(self.label_analyzerList)
        self.hlay_analyzerList.addWidget(self.comboBox_analyzer)

        self.label_visaString = QtWidgets.QLabel("Visa String", self)
        self.lineEdit_visaString = QtWidgets.QLineEdit(self)

        self.hlay_row1 = QtWidgets.QHBoxLayout()
        self.hlay_row1.addWidget(self.checkBox_TriggerNew)
        self.hlay_row1.addLayout(self.hlay_analyzerList)
        self.hlay_row1.addWidget(self.label_visaString)
        self.hlay_row1.addWidget(self.lineEdit_visaString)

        self.label_channel = QtWidgets.QLabel("Channel:")
        self.spinBox_channel = QtWidgets.QSpinBox()
        self.spinBox_channel.setMinimum(1)
        self.spinBox_channel.setMaximum(256)

        self.label_startFreq = QtWidgets.QLabel("Start Freq:", self)
        self.lineEdit_startFrequency = QtWidgets.QLineEdit(self)
        self.lineEdit_startFrequency.setValidator(QtGui.QDoubleValidator())
        self.lineEdit_startFrequency.setText("{:g}".format(0.01))
        self.lineEdit_startFrequency.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)

        self.label_stopFreq = QtWidgets.QLabel("Stop Freq:", self)
        self.lineEdit_stopFrequency = QtWidgets.QLineEdit(self)
        self.lineEdit_stopFrequency.setValidator(QtGui.QDoubleValidator())
        self.lineEdit_stopFrequency.setText("{:g}".format(40.0))
        self.lineEdit_stopFrequency.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)

        self.label_numberOfPoints = QtWidgets.QLabel("Num Points:", self)
        self.spinBox_numberOfPoints = QtWidgets.QSpinBox(self)
        self.spinBox_numberOfPoints.setMinimum(1)
        self.spinBox_numberOfPoints.setMaximum(100000)
        self.spinBox_numberOfPoints.setSingleStep(100)
        self.spinBox_numberOfPoints.setValue(401)

        self.label_funit = QtWidgets.QLabel("Units:", self)
        self.comboBox_funit = QtWidgets.QComboBox(self)
        self.comboBox_funit.addItems(self.FUNITS)
        self.comboBox_funit.setCurrentIndex(self.comboBox_funit.findText("GHz"))

        self.btn_setAnalyzerFreqSweep = QtWidgets.QPushButton("Set Freq. Sweep", self)

        self.layout_row2 = QtWidgets.QHBoxLayout()
        for label, widget in (
                (self.label_channel, self.spinBox_channel),
                (self.label_startFreq, self.lineEdit_startFrequency),
                (self.label_stopFreq, self.lineEdit_stopFrequency),
                (self.label_numberOfPoints, self.spinBox_numberOfPoints),
                (self.label_funit, self.comboBox_funit)
        ):
            label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
            self.layout_row2.addWidget(label)
            self.layout_row2.addWidget(widget)
        self.layout_row2.addWidget(self.btn_setAnalyzerFreqSweep)
        self.layout_row2.insertStretch(-1)

        self.verticalLayout.addLayout(self.hlay_row1)
        self.verticalLayout.addLayout(self.layout_row2)

        self.comboBox_analyzer.currentIndexChanged.connect(self.set_analyzer_default_address)
        for key, val in analyzers.items():
            self.comboBox_analyzer.addItem(key)
        # --- End Setup UI Elements --- #

        self._start_frequency = float(self.lineEdit_startFrequency.text())
        self._stop_frequency = float(self.lineEdit_stopFrequency.text())
        self.funit = self.comboBox_funit.currentText()
        self.comboBox_funit.currentIndexChanged.connect(self.frequency_changed)

        self.btn_setAnalyzerFreqSweep.clicked.connect(self.set_frequency_sweep)

    def set_frequency_sweep(self):
        channel = self.spinBox_channel.value()
        f_unit = self.comboBox_funit.currentText()
        f_start = self.start_frequency
        f_stop = self.stop_frequency
        f_npoints = self.spinBox_numberOfPoints.value()

        with self.get_analyzer() as nwa:
            nwa.set_frequency_sweep(channel=channel, f_unit=f_unit, f_start=f_start, f_stop=f_stop, f_npoints=f_npoints)

    def set_start_freequency(self, value):
        self._start_frequency = float(value)
        self.lineEdit_startFrequency.setText("{:g}".format(self._start_frequency))
    
    def get_start_frequency(self):
        self._start_frequency = float(self.lineEdit_startFrequency.text())
        return self._start_frequency
    
    start_frequency = property(get_start_frequency, set_start_freequency)

    def set_stop_freequency(self, value):
        self._stop_frequency = float(value)
        self.lineEdit_stopFrequency.setText("{:g}".format(self._stop_frequency))
    
    def get_stop_frequency(self):
        self._stop_frequency = float(self.lineEdit_stopFrequency.text())
        return self._stop_frequency
    
    stop_frequency = property(get_stop_frequency, set_stop_freequency)

    def frequency_changed(self):
        conversion = self.FCONVERSIONS[self.comboBox_funit.currentText()] / self.FCONVERSIONS[self.funit]
        self.funit = self.comboBox_funit.currentText()
        self.start_frequency *= conversion
        self.stop_frequency *= conversion

    def set_analyzer_default_address(self):
        self.lineEdit_visaString.setText(analyzers[self.comboBox_analyzer.currentText()].DEFAULT_VISA_ADDRESS)

    def get_analyzer(self):
        return analyzers[self.comboBox_analyzer.currentText()](self.lineEdit_visaString.text())


class ReflectDialog(QtWidgets.QDialog):
    def __init__(self, analyzer=None, parent=None):
        super(ReflectDialog, self).__init__(parent)

        self.setWindowTitle("Measure Reflect Standards")

        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.gridLayout = QtWidgets.QGridLayout()
        self.label_port2 = QtWidgets.QLabel(self)
        self.gridLayout.addWidget(self.label_port2, 1, 2, 1, 1)
        self.btn_loadPort1 = QtWidgets.QPushButton(self)
        self.gridLayout.addWidget(self.btn_loadPort1, 0, 1, 1, 1)
        self.btn_loadPort2 = QtWidgets.QPushButton(self)
        self.gridLayout.addWidget(self.btn_loadPort2, 1, 1, 1, 1)
        self.label_port1 = QtWidgets.QLabel(self)
        self.gridLayout.addWidget(self.label_port1, 0, 2, 1, 1)
        self.btn_measurePort2 = QtWidgets.QPushButton(self)
        self.gridLayout.addWidget(self.btn_measurePort2, 1, 0, 1, 1)
        self.btn_measurePort1 = QtWidgets.QPushButton(self)
        self.gridLayout.addWidget(self.btn_measurePort1, 0, 0, 1, 1)
        self.btn_measureBoth = QtWidgets.QPushButton(self)
        self.gridLayout.addWidget(self.btn_measureBoth, 2, 0, 1, 1)
        self.btn_loadBoth = QtWidgets.QPushButton(self)
        self.gridLayout.addWidget(self.btn_loadBoth, 2, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.verticalLayout.addWidget(self.buttonBox)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.label_port2.setText("port2 - not ready")
        self.btn_loadPort1.setText("Load Port 1 (.s1p)")
        self.btn_loadPort2.setText("Load Port (.s1p)")
        self.label_port1.setText("port1 - not ready")
        self.btn_measurePort2.setText("Measure Port2")
        self.btn_measurePort1.setText("Measure Port1")
        self.btn_measureBoth.setText("Measure Both")
        self.btn_loadBoth.setText("Load Both (.s2p)")

        self._ready = False
        self.analyzer = analyzer

        if self.analyzer is None:
            for btn in (self.btn_measureBoth, self.btn_measurePort1, self.btn_measurePort2):
                btn.setEnabled(False)

        self.reflect_2port = None
        self.s11 = None
        self.s22 = None

        self.ok = self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok)  # type: QtWidgets.QPushButton
        self.ok.setEnabled(False)

        self.btn_measureBoth.clicked.connect(self.measure_both)
        self.btn_measurePort1.clicked.connect(self.measure_s11)
        self.btn_measurePort2.clicked.connect(self.measure_s22)

        self.btn_loadBoth.clicked.connect(self.load_both)
        self.btn_loadPort1.clicked.connect(self.load_s11)
        self.btn_loadPort2.clicked.connect(self.load_s22)

    def measure_s11(self):
        self.s11 = self.analyzer.get_oneport(port=1)
        self.evaluate()

    def measure_s22(self):
        self.s22 = self.analyzer.get_oneport(port=2)
        self.evaluate()

    def measure_both(self):
        self.reflect_2port = self.analyzer.get_twoport()
        self.evaluate()

    def load_s11(self):
        self.s11 = load_network_file("load port 1 reflect", "1-port touchstone (*.s1p)")
        self.evaluate()

    def load_s22(self):
        self.s22 = load_network_file("load port 2 reflect", "1-port touchstone (*.s1p)")
        self.evaluate()

    def load_both(self):
        self.reflect_2port = load_network_file("load reflect cal standard")
        self.evaluate()

    @property
    def ready(self): return self._ready

    @ready.setter
    def ready(self, val):
        if val is True:
            self._ready = True
            self.ok.setEnabled(True)
        else:
            self._ready = False
            self.ok.setEnabled(False)

    def evaluate(self):
        if type(self.reflect_2port) is skrf.Network:
            self.ready = True
            self.label_port1.setText("port1 - measured")
            self.label_port2.setText("port2 - measured")
        else:
            if type(self.s11) is skrf.Network and type(self.s22) is skrf.Network:
                self.reflect_2port = skrf.two_port_reflect(self.s11, self.s22)
                # self.reflect_2port = skrf.four_oneports_2_twoport(self.s11, self.s11, self.s22, self.s22)
                # self.reflect_2port.s[:, 0, 1] = 0
                # self.reflect_2port.s[:, 1, 0] = 0
                self.ready = True
                self.label_port1.setText("port1 - measured")
                self.label_port2.setText("port2 - measured")
            else:
                self.ready = False
                if type(self.s11) is skrf.Network:
                    self.label_port1.setText("port1 - measured")
                else:
                    self.label_port1.setText("port1 - not measured")
                if type(self.s22) is skrf.Network:
                    self.label_port2.setText("port2 - measured")
                else:
                    self.label_port2.setText("port2 - not measured")


class MeasurementDialog(QtWidgets.QDialog):

    measurements_available = QtCore.Signal(object)

    def __init__(self, nwa, parent=None):
        super(MeasurementDialog, self).__init__(parent)
        self.setWindowTitle("MeasurementDialog")
        self.horizontalLayout_main = QtWidgets.QHBoxLayout(self)

        self.verticalLayout_left = QtWidgets.QVBoxLayout()

        self.groupBox_options = QtWidgets.QGroupBox("Options", self)
        self.lineEdit_namePrefix = QtWidgets.QLineEdit(self)
        self.label_namePrefix = QtWidgets.QLabel("Name Prefix:")
        self.horizontalLayout_namePrefix = QtWidgets.QHBoxLayout()
        self.horizontalLayout_namePrefix.addWidget(self.label_namePrefix)
        self.horizontalLayout_namePrefix.addWidget(self.lineEdit_namePrefix)
        self.label_timeout = QtWidgets.QLabel("Timeout (ms)", self)
        self.spinBox_timeout = QtWidgets.QSpinBox(self)
        self.spinBox_timeout.setMinimum(100)
        self.spinBox_timeout.setMaximum(600000)
        try:
            self.spinBox_timeout.setValue(nwa.resource.timeout)
        except:
            self.spinBox_timeout.setValue(3000)
        self.spinBox_timeout.setSingleStep(1000)
        self.horizontalLayout_timeout = QtWidgets.QHBoxLayout()
        self.horizontalLayout_timeout.addWidget(self.label_timeout)
        self.horizontalLayout_timeout.addWidget(self.spinBox_timeout)
        self.checkBox_sweepNew = QtWidgets.QCheckBox("Sweep New", self.groupBox_options)
        self.checkBox_autoTimeOut = QtWidgets.QCheckBox("Auto Timeout", self.groupBox_options)
        self.horizonatlLayout_sweep = QtWidgets.QHBoxLayout()
        self.horizonatlLayout_sweep.addWidget(self.checkBox_sweepNew)
        self.horizonatlLayout_sweep.addWidget(self.checkBox_autoTimeOut)
        self.label_channel = QtWidgets.QLabel("Channel", self.groupBox_options)
        self.spinBox_channel = QtWidgets.QSpinBox(self.groupBox_options)
        self.horizontalLayout_channel = QtWidgets.QHBoxLayout()
        self.horizontalLayout_channel.addWidget(self.label_channel)
        self.horizontalLayout_channel.addWidget(self.spinBox_channel)

        self.verticalLayout_options = QtWidgets.QVBoxLayout(self.groupBox_options)
        self.verticalLayout_options.addLayout(self.horizontalLayout_namePrefix)
        self.verticalLayout_options.addLayout(self.horizontalLayout_timeout)
        self.verticalLayout_options.addLayout(self.horizonatlLayout_sweep)
        self.verticalLayout_options.addLayout(self.horizontalLayout_channel)
        self.verticalLayout_left.addWidget(self.groupBox_options)

        self.groupBox_snp = QtWidgets.QGroupBox("Get N-Port Network", self)
        self.verticalLayout_snp = QtWidgets.QVBoxLayout(self.groupBox_snp)
        self.label_ports = QtWidgets.QLabel("Ports:", self.groupBox_snp)
        self.lineEdit_ports = QtWidgets.QLineEdit(self.groupBox_snp)
        self.btn_measureSnp = QtWidgets.QPushButton("Measure Network", self.groupBox_snp)
        self.horizontalLayout_nports = QtWidgets.QHBoxLayout()
        self.horizontalLayout_nports.addWidget(self.label_ports)
        self.horizontalLayout_nports.addWidget(self.lineEdit_ports)
        self.verticalLayout_snp.addWidget(self.btn_measureSnp)
        self.verticalLayout_snp.addLayout(self.horizontalLayout_nports)

        self.spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_snp.addItem(self.spacerItem)
        self.verticalLayout_left.addWidget(self.groupBox_snp)

        self.groupBox_traces = QtWidgets.QGroupBox("Available Traces", self)
        self.listWidget_traces = QtWidgets.QListWidget(self.groupBox_traces)
        self.listWidget_traces.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.btn_updateTraces = QtWidgets.QPushButton("Update", self.groupBox_traces)
        self.btn_measureTraces = QtWidgets.QPushButton("Measure Traces", self.groupBox_traces)
        self.horizontalLayout_tracesButtons = QtWidgets.QHBoxLayout()
        self.horizontalLayout_tracesButtons.addWidget(self.btn_updateTraces)
        self.horizontalLayout_tracesButtons.addWidget(self.btn_measureTraces)

        self.verticalLayout_traces = QtWidgets.QVBoxLayout(self.groupBox_traces)
        self.verticalLayout_traces.addWidget(self.listWidget_traces)
        self.verticalLayout_traces.addLayout(self.horizontalLayout_tracesButtons)

        self.horizontalLayout_main.addLayout(self.verticalLayout_left)
        self.horizontalLayout_main.addWidget(self.groupBox_traces)

        self.nwa = nwa

        self.btn_updateTraces.clicked.connect(self.update_traces)
        self.btn_measureSnp.clicked.connect(self.measure_snp)
        self.btn_measureTraces.clicked.connect(self.measure_traces)

        if self.nwa.NCHANNELS:
            self.spinBox_channel.setValue(1)
            self.spinBox_channel.setMinimum(1)
            self.spinBox_channel.setMaximum(self.nwa.NCHANNELS)
        else:
            self.spinBox_channel.setEnabled(False)

        self.lineEdit_ports.setText(",".join([str(port+1) for port in range(self.nwa.nports)]))
        self.spinBox_timeout.valueChanged.connect(self.set_timeout)

    def set_timeout(self):
        self.nwa.resource.timeout = self.spinBox_timeout.value()

    def measure_traces(self):
        items = self.listWidget_traces.selectedItems()
        if len(items) < 1:
            print("nothing to measure")
            return

        traces = []
        for item in items:
            traces.append(item.trace)

        ntwks = self.nwa.get_traces(traces, name_prefix=self.lineEdit_namePrefix.text())
        self.measurements_available.emit(ntwks)

    def measure_snp(self):
        ports = self.lineEdit_ports.text().replace(" ", "").split(",")
        try:
            ports = [int(port) for port in ports]
        except Exception:
            qt.error_popup("Ports must be a comma separated list of integers")
            return

        kwargs = {"ports": ports,
                  "channel": self.spinBox_channel.value(),
                  "sweep": self.checkBox_sweepNew.isChecked(),
                  "name": self.lineEdit_namePrefix.text()}
        if self.checkBox_autoTimeOut.isChecked():
            kwargs["timeout"] = self.spinBox_timeout.value()

        ntwk = self.nwa.get_snp_network(**kwargs)
        self.measurements_available.emit(ntwk)

    def update_traces(self):
        traces = self.nwa.get_list_of_traces()
        self.listWidget_traces.clear()
        for trace in traces:
            item = QtWidgets.QListWidgetItem()
            item.setText(trace["label"])
            item.trace = trace
            self.listWidget_traces.addItem(item)
