import sys
import os
import traceback
import re
from collections import OrderedDict
from math import sqrt

import numpy as np
import sip
from qtpy import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import skrf

from . import qt
from . import smith_chart
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


def trace_color_cycle(n=1000):
    """
    :type n: int
    :return:
    """

    lime_green = "#00FF00"
    cyan = "#00FFFF"
    magenta = "#FF00FF"
    yellow = "#FFFF00"
    pink = "#C04040"
    blue = "#0000FF"
    lavendar = "#FF40FF"
    turquoise = "#00FFFF"

    count = 0
    colors = [yellow, cyan, magenta, lime_green, pink, blue, lavendar, turquoise]
    num = len(colors)
    while count < n:
        yield colors[count % num]
        count += 1


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


@QtCore.Slot(object, str)
def save_NetworkListItem(ntwk_list_item, save_which):
    """
    :type ntwk_list_item: NetworkListItem
    :type save_which: str
    :return:
    """

    if save_which.lower() not in ("raw", "cal", "both"):
        raise ValueError("Must set save option to 'raw', 'cal', or 'both'")

    ntwk = ntwk_list_item.ntwk
    ntwk_c = ntwk_list_item.ntwk_corrected

    if type(ntwk) in (list, tuple):
        save_multiple_networks(ntwk)
        return
    elif not isinstance(ntwk, skrf.Network):
        raise TypeError("ntwk must be a skrf.Network object to save")

    extension = ".s{:d}p".format(ntwk.s.shape[1])
    file_filter = "touchstone format (*{:s})".format(extension)
    filename = os.path.join(qt.cfg.last_path, ntwk.name + extension)

    if save_which.lower() == "both":
        if isinstance(ntwk, skrf.Network) and isinstance(ntwk_c, skrf.Network):
            filename = qt.getSaveFileName_Global("Save Raw skrf.Network File", filter=file_filter, start_path=filename)
            if not filename:
                return
            base, ext = os.path.splitext(filename)
            filename_cal = base + "-cal" + ext
            filename_cal = qt.getSaveFileName_Global("Save Calibrated skrf.Network File",
                                                 filter=file_filter,
                                                 start_path=filename_cal)
            if not filename_cal:
                return
            ntwk.write_touchstone(filename)
            ntwk_c.write_touchstone(filename_cal)
            return
        else:
            save_which = "raw"

    if save_which.lower() == "cal":
        if ntwk_list_item.ntwk_corrected is None:
            Warning("ntwk_corrected is None, saving raw instead")
            save_which = "raw"
        else:
            ntwk = ntwk_list_item.ntwk_corrected
            filename = os.path.join(qt.cfg.last_path, ntwk.name + extension)
            if not filename:
                return

    caption = "Save skrf.Network File" if save_which.lower() == "raw" else "Save Calibrated skrf.Network File"
    filename = qt.getSaveFileName_Global(caption, filter=file_filter, start_path=filename)
    if not filename:
        return
    ntwk.write_touchstone(filename)


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


class NetworkListItem(QtWidgets.QListWidgetItem):

    def __init__(self, parent=None):
        super(NetworkListItem, self).__init__(parent)
        self.ntwk = None
        self.ntwk_corrected = None

    def update_ntwk_names(self):
        if isinstance(self.ntwk, skrf.Network):
            self.ntwk.name = self.text()
        if isinstance(self.ntwk_corrected, skrf.Network):
            self.ntwk_corrected.name = self.text() + "-cal"


class NetworkListWidget(QtWidgets.QListWidget):
    MEASUREMENT_PREFIX = "meas"

    item_removed = QtCore.Signal()
    item_updated = QtCore.Signal(object)
    save_single_requested = QtCore.Signal(object, str)

    def __init__(self, ntwk_plot=None, name_prefix='meas', parent=None):
        super(NetworkListWidget, self).__init__(parent)
        self.name_prefix = name_prefix
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.customContextMenuRequested.connect(self.list_item_right_clicked)
        self.save_single_requested.connect(save_NetworkListItem)
        self.itemDelegate().commitData.connect(self.item_text_updated)
        self.itemSelectionChanged.connect(self.set_active_networks)
        self.item_updated.connect(self.set_active_network)

        self._ntwk_plot = None
        self.named_items = {}

    @property
    def ntwk_plot(self):
        return self._ntwk_plot  # type: NetworkPlotWidget

    @ntwk_plot.setter
    def ntwk_plot(self, ntwk_plot):
        if isinstance(ntwk_plot, NetworkPlotWidget):
            self._ntwk_plot = ntwk_plot
            self.item_removed.connect(self._ntwk_plot.clear_plot)
        else:
            self._ntwk_plot = None
            self.item_removed.disconnect()

    def get_unique_name(self, name=None, exclude_item=-1):
        """
        :type name: str
        :type exclude_item: int
        :return:
        """
        names = []
        for i in range(self.count()):
            if i == exclude_item:
                continue

            item = self.item(i)
            names.append(item.text())

        if name in names:
            if re.match("_\d\d", name[-3:]):
                name_base = name[:-3]
                suffix = int(name[-2:])
            else:
                name_base = name
                suffix = 1

            for num in range(suffix, 100, 1):
                name = "{:s}_{:02d}".format(name_base, num)
                if name not in names:
                    break
        return name

    def item_text_updated(self, emit=True):
        item = self.currentItem()  # type: NetworkListItem
        item.setText(self.get_unique_name(item.text(), self.row(item)))
        item.update_ntwk_names()
        if emit:
            self.item_updated.emit(item)

    def list_item_right_clicked(self, position):
        menu = QtWidgets.QMenu()

        if len(self.selectedItems()) == 1:
            save = QtWidgets.QAction("Save Item", self)
            menu.addAction(save)
            save.triggered.connect(self.save_selected_items)

            remove = QtWidgets.QAction("Remove Item", self)
            menu.addAction(remove)
            remove.triggered.connect(self.remove_item)
        elif len(self.selectedItems()) > 1:
            save = QtWidgets.QAction("Save Items", self)
            menu.addAction(save)
            save.triggered.connect(self.save_selected_items)

            remove = QtWidgets.QAction("Remove Items", self)
            menu.addAction(remove)
            remove.triggered.connect(self.remove_item)

        menu.exec_(self.mapToGlobal(position))  # QtWidgets.QAction

    def remove_item(self):
        items = self.selectedItems()
        if len(items) > 0:
            self.item_removed.emit()
            for item in items:
                self.takeItem(self.row(item))

    def get_save_which_mode(self):
        """
        because the item will potentially have a raw and calibrated network attached we need
        to determine if we want to save
        "raw", "cal", or "both"
        the default will be to save raw, and this method must be replaced by the parent infrastructure
        for a different result

        :return: str
        """
        return "raw"

    def set_active_networks(self):
        if not self.ntwk_plot:
            return

        items = self.selectedItems()        
        ntwk_list = []
        ntwk_list_corrected = []

        for item in items:
            if isinstance(item.ntwk, skrf.Network):
                ntwk_list.append(item.ntwk)
            elif type(item.ntwk) in (list, tuple):
                ntwk_list.extend(item.ntwk)
                
            if isinstance(item.ntwk_corrected, skrf.Network):
                ntwk_list_corrected.append(item.ntwk_corrected)
            elif type(item.ntwk_corrected) in (list, tuple):
                ntwk_list_corrected.extend(item.ntwk_corrected)
        
        if ntwk_list:
            ntwk = ntwk_list if len(ntwk_list) > 1 else ntwk_list[0]
        else:
            ntwk = None

        if ntwk_list_corrected:
            ntwk_corrected = ntwk_list_corrected if len(ntwk_list_corrected) > 1 else ntwk_list_corrected[0]
        else:
            ntwk_corrected = None

        self.ntwk_plot.set_networks(ntwk, ntwk_corrected)

    def set_active_network(self, item):
        """
        :type item: NetworkListItem
        :return:
        """
        if item is None:
            return

        if self.ntwk_plot:
            self.ntwk_plot.set_networks(item.ntwk, item.ntwk_corrected)

    def load_named_ntwk(self, ntwk, name, activate=True):
        item = self.get_named_item(name)
        if not item:
            item = NetworkListItem()
            self.addItem(item)

        item.setFlags(item.flags() | QtCore.Qt.NoItemFlags)
        item.setText(name)
        item.ntwk = ntwk
        self.clearSelection()
        self.setCurrentItem(item)
        if activate is True:
            self.set_active_network(item)

    def get_named_item(self, name):
        items = self.findItems(name, QtCore.Qt.MatchExactly)
        if len(items) > 1:
            raise Exception("multiple items matched name: {:s}".format(name))
        elif len(items) == 0:
            item = None
        else:
            item = items[0]
        return item

    def load_network(self, ntwk, activate=True):
        item = NetworkListItem()
        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        item.setText(ntwk.name)
        item.ntwk = ntwk
        self.addItem(item)
        self.clearSelection()
        self.setCurrentItem(item)
        self.item_text_updated(emit=activate)
        return item

    def load_from_file(self, caption="load touchstone file"):
        ntwk = load_network_file(caption)  # type: skrf.Network
        if not ntwk:
            return
        self.load_network(ntwk)

    def load_from_files(self, caption="load touchstone file"):
        ntwks = load_network_files(caption)  # type: skrf.Network
        if not ntwks:
            return

        try:
            self.blockSignals(False)
            for ntwk in ntwks:
                self.load_network(ntwk, False)
        finally:
            self.blockSignals(False)

        item = self.item(self.count() - 1)
        self.set_active_network(item)

    def save_selected_items(self):
        items = self.selectedItems()
        if len(items) == 1:
            item = items[0]
            self.save_single_requested.emit(item, self.get_save_which_mode())
        elif len(items) > 1:
            ntwk_list = []
            for item in items:
                ntwk_list.append(item.ntwk)
            save_multiple_networks(ntwk_list)

    def save_all_measurements(self):
        save_which = self.get_save_which_mode()
        ntwk_list = []

        for i in range(self.count()):
            item = self.item(i)
            if save_which != "cal" and isinstance(item.ntwk, skrf.Network):
                ntwk_list.append(item.ntwk)
            if save_which != "raw" and isinstance(item.ntwk_corrected, skrf.Network):
                ntwk_list.append(item.ntwk_corrected)

        save_multiple_networks(ntwk_list)

    def get_analyzer(self):
        raise AttributeError("Must set get_analyzer method externally")

    def get_all_networks(self, corrected=False):
        ntwks = []
        for i in range(self.count()):
            item = self.item(i)
            ntwk = item.ntwk_corrected if corrected is True else item.ntwk
            if isinstance(ntwk, skrf.Network):
                ntwks.append(ntwk)
            elif type(ntwk) in (list, tuple):
                ntwks.extend(ntwk)
        return ntwks

    def measure_ntwk(self):
        with self.get_analyzer() as nwa:
            meas = nwa.measure_twoport_ntwk()
            meas.name = self.MEASUREMENT_PREFIX  # unique name processed in load_network
        self.load_network(meas)

    def get_load_button(self, label="Load"):
        button = QtWidgets.QPushButton(label)
        button.released.connect(self.load_from_files)
        return button

    def get_measure_button(self, label="Measure"):
        button = QtWidgets.QPushButton(label)
        button.released.connect(self.measure_ntwk)
        return button

    def get_save_selected_button(self, label="Save Selected"):
        button = QtWidgets.QPushButton(label)
        button.clicked.connect(self.save_selected_items)
        return button

    def get_save_all_button(self, label="Save All"):
        button = QtWidgets.QPushButton(label)
        button.clicked.connect(self.save_all_measurements)
        return button

    def get_input_buttons(self, labels=("Load", "Measure")):
        widget = QtWidgets.QWidget()
        horizontal_layout = QtWidgets.QHBoxLayout(widget)
        horizontal_layout.setContentsMargins(0, 2, 0, 2)
        load_button = self.get_load_button(labels[0])
        measurement_button = self.get_measure_button(labels[1])
        horizontal_layout.addWidget(load_button)
        horizontal_layout.addWidget(measurement_button)
        return widget

    def get_save_buttons(self, labels=("Save Selected", "Save All")):
        widget = QtWidgets.QWidget()
        horizontal_layout = QtWidgets.QHBoxLayout(widget)
        horizontal_layout.setContentsMargins(0, 2, 0, 2)
        save_selected_button = self.get_save_selected_button(labels[0])
        save_all_button = self.get_save_all_button(labels[1])
        horizontal_layout.addWidget(save_selected_button)
        horizontal_layout.addWidget(save_all_button)
        return widget


class NetworkPlotWidget(QtWidgets.QWidget):
    S_VALS = OrderedDict((
        ("decibels", "db"),
        ("magnitude", "mag"),
        ("phase (deg)", "deg"),
        ("phase unwrapped (deg)", "deg_unwrap"),
        ("phase (rad)", "rad"),
        ("phase unwrapped (rad)", "rad_unwrap"),
        ("real", "re"),
        ("imaginary", "im"),
    ))
    S_UNITS = list(S_VALS.keys())

    def __init__(self, parent=None, **kwargs):
        super(NetworkPlotWidget, self).__init__(parent)

        self.checkBox_useCorrected = QtWidgets.QCheckBox()
        self.checkBox_useCorrected.setText("Plot Corrected")

        self.comboBox_primarySelector = QtWidgets.QComboBox(self)
        self.comboBox_primarySelector.addItems(("S", "Z", "Y", "A", "Smith Chart"))

        self.comboBox_unitsSelector = QtWidgets.QComboBox(self)
        self.comboBox_unitsSelector.addItems(self.S_UNITS)

        self.comboBox_traceSelector = QtWidgets.QComboBox(self)
        self.set_trace_items()
        self.comboBox_traceSelector.setCurrentIndex(0)

        self.plot_layout = pg.GraphicsLayoutWidget(self)
        self.plot_layout.sceneObj.sigMouseClicked.connect(self.graph_clicked)

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.addWidget(self.checkBox_useCorrected)
        self.horizontalLayout.addWidget(self.comboBox_primarySelector)
        self.horizontalLayout.addWidget(self.comboBox_unitsSelector)
        self.horizontalLayout.addWidget(self.comboBox_traceSelector)

        self.data_info_label = QtWidgets.QLabel("Click a data point to see info")

        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setContentsMargins(3, 3, 3, 3)  # normally this will be embedded in another application
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout.addWidget(self.plot_layout)
        self.verticalLayout.addWidget(self.data_info_label)

        self.checkBox_useCorrected.stateChanged.connect(self.set_use_corrected)
        self.comboBox_primarySelector.currentIndexChanged.connect(self.update_plot)
        self.comboBox_unitsSelector.currentIndexChanged.connect(self.update_plot)
        self.comboBox_traceSelector.currentIndexChanged.connect(self.update_plot)
        
        self.plot = self.plot_layout.addPlot()  # type: pg.PlotItem

        self._ntwk = None
        self._ntwk_corrected = None
        self._corrected_data_enabled = True
        self._use_corrected = False
        self.corrected_data_enabled = kwargs.get('corrected_data_enabled', True)

        self.plot.addLegend()
        self.plot.showGrid(True, True)
        self.plot.setLabel("bottom", "frequency", units="Hz")

        self.last_plot = "rectangular"

    def get_use_corrected(self):
        return self._use_corrected

    def set_use_corrected(self, val):
        if val in (1, 2):
            self._use_corrected = True
        else:
            self._use_corrected = False
        self.update_plot()

    use_corrected = property(get_use_corrected, set_use_corrected)

    @property
    def ntwk(self): return self._ntwk

    @ntwk.setter
    def ntwk(self, ntwk):
        if ntwk is None or isinstance(ntwk, skrf.Network) or type(ntwk) in (list, tuple):
            self.set_trace_items(ntwk)
            self._ntwk = ntwk
            self.update_plot()
        else:
            raise TypeError("must set to skrf.Network, list of Networks, or None")

    @property
    def ntwk_corrected(self): return self._ntwk_corrected

    @ntwk_corrected.setter
    def ntwk_corrected(self, ntwk):
        if ntwk is None or isinstance(ntwk, skrf.Network) or type(ntwk) in (list, tuple):
            self.set_trace_items(ntwk)
            self._ntwk_corrected = ntwk   
            self.update_plot()
        else:
            raise TypeError("must set to skrf.Network, list of Networks, or None")

    @property
    def corrected_data_enabled(self):
        return self._corrected_data_enabled

    @corrected_data_enabled.setter
    def corrected_data_enabled(self, enabled):
        if enabled is True:
            self._corrected_data_enabled = True
            self.checkBox_useCorrected.setEnabled(True)
        else:
            self._corrected_data_enabled = False
            self._use_corrected = False
            self.checkBox_useCorrected.setEnabled(False)

    def set_networks(self, ntwk, ntwk_corrected=None):
        if ntwk is None or isinstance(ntwk, skrf.Network) or type(ntwk) in (list, tuple):
            self._ntwk = ntwk
            self.set_trace_items(self._ntwk)
            if ntwk is None:
                self._ntwk_corrected = None
                self.set_trace_items(self._ntwk)
                return
        else:
            raise TypeError("must set to skrf.Network, list of Networks, or None")

        if ntwk_corrected is None or isinstance(ntwk_corrected, skrf.Network) or type(ntwk_corrected) in (list, tuple):
            self._ntwk_corrected = ntwk_corrected
        else:
            raise TypeError("must set to skrf.Network, list of Networks, or None")

        self.update_plot()

    def reset_plot(self, smith=False):
        self.plot.clear()

        if not smith and self.last_plot == "smith":
            self.plot.setAspectLocked(False)
            self.plot.autoRange()
            self.plot.enableAutoRange()
            self.plot.setLabel("bottom", "frequency", units="Hz")

        if smith and not self.last_plot == "smith":
            self.last_plot = "smith"
            self.ZGrid = smith_chart.gen_z_grid()
            self.s_unity_circle = smith_chart.gen_s_unity_circle()
            self.plot_layout.removeItem(self.plot)
            self.plot = self.plot_layout.addPlot()
            self.plot.setAspectLocked()
            self.plot.setXRange(-1, 1)
            self.plot.setYRange(-1, 1)

        if smith:
            self.plot.addItem(self.s_unity_circle)
            self.plot.addItem(self.ZGrid)

        if not smith:
            self.plot.setLabel("left", "")

        self.plot.setTitle(None)
        legend = self.plot.legend
        if legend is not None:
            legend.scene().removeItem(legend)
        self.plot.addLegend()

    def clear_plot(self):
        self._ntwk = None
        self._ntwk_corrected = None
        self._ntwk_list = None
        self.reset_plot()

    def set_trace_items(self, ntwk=None):
        self.comboBox_traceSelector.blockSignals(True)
        current_index = self.comboBox_traceSelector.currentIndex()
        nports = 0

        if isinstance(ntwk, skrf.Network):
            nports = ntwk.nports
        elif type(ntwk) in (list, tuple):
            for n in ntwk:
                if n.nports > nports:
                    nports = n.nports

        self.comboBox_traceSelector.clear()
        self.comboBox_traceSelector.addItem("all")

        for n in range(nports):
            for m in range(nports):
                self.comboBox_traceSelector.addItem("S{:d}{:d}".format(m + 1, n + 1))

        if current_index <= self.comboBox_traceSelector.count():
            self.comboBox_traceSelector.setCurrentIndex(current_index)
        else:
            self.comboBox_traceSelector.setCurrentIndex(0)
        self.comboBox_traceSelector.blockSignals(False)

    def graph_clicked(self, ev):
        """
        :type ev: pg.GraphicsScene.mouseEvents.MouseClickEvent
        :return:
        """
        xy = self.plot.vb.mapSceneToView(ev.scenePos())
        if not ev.isAccepted():
            if "smith" in self.comboBox_primarySelector.currentText().lower():
                S11 = xy.x() + 1j * xy.y()
                Z = (1 + S11) / (1 - S11)
                self.data_info_label.setText(
                    "Sre: {:g}, Sim: {:g}  -  R: {:g}, X: {:g}".format(xy.x(), xy.y(), Z.real, Z.imag))
            else:
                self.data_info_label.setText("x: {:g}, y: {:g}".format(xy.x(), xy.y()))
        elif isinstance(ev.acceptedItem, pg.PlotCurveItem):
            curve = ev.acceptedItem  # type: pg.PlotCurveItem
            spoint = xy.x() + 1j * xy.y()
            sdata = curve.xData + 1j * curve.yData
            index = np.argmin(np.abs(sdata - spoint))
            frequency = curve.ntwk.frequency.f_scaled[index]
            S11 = curve.xData[index] + 1j * curve.yData[index]
            Z = (1 + S11) / (1 - S11)
            self.data_info_label.setText(
                "Freq: {:g} ({:s}), S(re): {:g}, S(im): {:g}  -  R: {:g}, X: {:g}".format(
                    frequency, curve.ntwk.frequency.unit, S11.real, S11.imag, Z.real, Z.imag))

    def update_plot(self):
        if "smith" in self.comboBox_primarySelector.currentText().lower():
            self.plot_smith()
        else:
            self.plot_ntwk()
            self.last_plot = "rectangular"

    def plot_ntwk(self):
        if self.use_corrected and self.ntwk_corrected is not None:
            ntwk = self.ntwk_corrected
        else:
            ntwk = self.ntwk

        if ntwk is None:
            return
        elif type(ntwk) in (list, tuple):
            self.plot_ntwk_list()
            return

        self.reset_plot()
        self.plot.showGrid(True, True)
        self.plot.setLabel("bottom", "frequency", units="Hz")

        colors = trace_color_cycle(ntwk.s.shape[1] ** 2)

        trace = self.comboBox_traceSelector.currentIndex()
        n_ = m_ = 0
        if trace > 0:
            mn = trace - 1
            nports = int(sqrt(self.comboBox_traceSelector.count() - 1))
            m_ = mn % nports
            n_ = int((mn - mn % nports) / nports)

        primary = self.comboBox_primarySelector.currentText().lower()
        s_units = self.comboBox_unitsSelector.currentText()
        attr = primary + "_" + self.S_VALS[s_units]
        s = getattr(ntwk, attr)
        for n in range(ntwk.s.shape[2]):
            for m in range(ntwk.s.shape[1]):
                c = next(colors)

                if trace > 0:
                    if not n == n_ or not m == m_:
                        continue

                label = "S{:d}{:d}".format(m + 1, n + 1)

                self.plot.plot(ntwk.f, s[:, m, n], pen=pg.mkPen(c), name=label)
        self.plot.setLabel("left", s_units)
        self.plot.setTitle(ntwk.name)

    def plot_ntwk_list(self):
        ntwk_list = self.ntwk
        if ntwk_list is None:
            return

        self.reset_plot()
        self.plot.showGrid(True, True)
        self.plot.setLabel("bottom", "frequency", units="Hz")

        colors = trace_color_cycle()

        trace = self.comboBox_traceSelector.currentIndex()
        n_ = m_ = 0
        if trace > 0:
            mn = trace - 1
            nports = int(sqrt(self.comboBox_traceSelector.count() - 1))
            m_ = mn % nports
            n_ = int((mn - mn % nports) / nports)

        primary = self.comboBox_primarySelector.currentText().lower()
        s_units = self.comboBox_unitsSelector.currentText()
        attr = primary + "_" + self.S_VALS[s_units]

        for ntwk in ntwk_list:
            s = getattr(ntwk, attr)
            for n in range(ntwk.s.shape[2]):
                for m in range(ntwk.s.shape[1]):
                    c = next(colors)

                    if trace > 0:
                        if not n == n_ or not m == m_:
                            continue

                    label = ntwk.name
                    if ntwk.s.shape[1] > 1:
                        label += " - S{:d}{:d}".format(m + 1, n + 1)

                    self.plot.plot(ntwk.f, s[:, m, n], pen=pg.mkPen(c), name=label)
        self.plot.setLabel("left", s_units)

    def plot_smith(self):
        if self.use_corrected and self.ntwk_corrected is not None:
            ntwk = self.ntwk_corrected
        else:
            ntwk = self.ntwk

        if ntwk is None:
            self.reset_plot(smith=True)
            return
        elif type(ntwk) in (list, tuple):
            self.plot_smith_list()
            return

        self.reset_plot(smith=True)

        colors = trace_color_cycle(ntwk.s.shape[1] ** 2)

        trace = self.comboBox_traceSelector.currentIndex()
        n_ = m_ = 0
        if trace > 0:
            mn = trace - 1
            nports = int(sqrt(self.comboBox_traceSelector.count() - 1))
            m_ = mn % nports
            n_ = int((mn - mn % nports) / nports)

        for n in range(ntwk.s.shape[2]):
            for m in range(ntwk.s.shape[1]):
                c = next(colors)

                if trace > 0:
                    if not n == n_ or not m == m_:
                        continue

                label = "S{:d}{:d}".format(m + 1, n + 1)

                s = ntwk.s[:, m, n]
                curve = self.plot.plot(s.real, s.imag, pen=pg.mkPen(c), name=label)
                curve.curve.setClickable(True)
                curve.curve.ntwk = ntwk
        self.plot.setTitle(ntwk.name)

    def plot_smith_list(self):
        self.reset_plot(smith=True)

        ntwk_list = self.ntwk
        if ntwk_list is None:
            return

        colors = trace_color_cycle()

        trace = self.comboBox_traceSelector.currentIndex()
        n_ = m_ = 0
        if trace > 0:
            mn = trace - 1
            nports = int(sqrt(self.comboBox_traceSelector.count() - 1))
            m_ = mn % nports
            n_ = int((mn - mn % nports) / nports)

        for ntwk in ntwk_list:
            for n in range(ntwk.s.shape[2]):
                for m in range(ntwk.s.shape[1]):
                    c = next(colors)

                    if trace > 0:
                        if not n == n_ or not m == m_:
                            continue

                    label = ntwk.name
                    if ntwk.s.shape[1] > 1:
                        label += " - S{:d}{:d}".format(m + 1, n + 1)

                    s = ntwk.s[:, m, n]
                    curve = self.plot.plot(s.real, s.imag, pen=pg.mkPen(c), name=label)
                    curve.curve.setClickable(True)
                    curve.curve.ntwk = ntwk


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
        self.forward, self.reverse = self.analyzer.measure_switch_terms()
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
        self.lineEdit_startFrequency.textEdited.connect(self.set_start_freequency)
        self.lineEdit_stopFrequency.textEdited.connect(self.set_stop_freequency)

    def set_start_freequency(self, value):
        self._start_frequency = float(value)
        self.lineEdit_startFrequency.setText("{:g}".format(self._start_frequency))
    
    def get_start_frequency(self):
        return self._start_frequency
    
    start_frequency = property(get_start_frequency, set_start_freequency)

    def set_stop_freequency(self, value):
        self._stop_frequency = float(value)
        self.lineEdit_stopFrequency.setText("{:g}".format(self._stop_frequency))
    
    def get_stop_frequency(self):
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
                self.reflect_2port = skrf.four_oneports_2_twoport(self.s11, self.s11, self.s22, self.s22)
                self.reflect_2port.s[:, 0, 1] = 0
                self.reflect_2port.s[:, 1, 0] = 0
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
        self.checkBox_sweepNew = QtWidgets.QCheckBox("Sweep New", self.groupBox_options)
        self.label_channel = QtWidgets.QLabel("Channel", self.groupBox_options)
        self.spinBox_channel = QtWidgets.QSpinBox(self.groupBox_options)
        self.horizontalLayout_channel = QtWidgets.QHBoxLayout()
        self.horizontalLayout_channel.addWidget(self.label_channel)
        self.horizontalLayout_channel.addWidget(self.spinBox_channel)

        self.verticalLayout_options = QtWidgets.QVBoxLayout(self.groupBox_options)
        self.verticalLayout_options.addWidget(self.checkBox_sweepNew)
        self.verticalLayout_options.addLayout(self.horizontalLayout_channel)
        self.verticalLayout_left.addWidget(self.groupBox_options)

        self.groupBox_snp = QtWidgets.QGroupBox("Get N-Port Network", self)
        self.verticalLayout_snp = QtWidgets.QVBoxLayout(self.groupBox_snp)
        self.label_nports = QtWidgets.QLabel("NPorts", self.groupBox_snp)
        self.comboBox_nports = QtWidgets.QComboBox(self.groupBox_snp)
        self.comboBox_nports.addItems(("1", "2"))
        self.btn_measureSnp = QtWidgets.QPushButton("Measure Network", self.groupBox_snp)
        self.horizontalLayout_nports = QtWidgets.QHBoxLayout()
        self.horizontalLayout_nports.addWidget(self.label_nports)
        self.horizontalLayout_nports.addWidget(self.comboBox_nports)
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

        self.snp_rows = list()
        self.nwa = nwa

        self.comboBox_nports.currentIndexChanged.connect(self.setup_nports)
        self.comboBox_nports.setCurrentIndex(1)

        self.btn_updateTraces.clicked.connect(self.update_traces)
        self.btn_measureSnp.clicked.connect(self.measure_snp_network)

        if self.nwa.NCHANNELS:
            self.spinBox_channel.setValue(1)
            self.spinBox_channel.setMinimum(1)
            self.spinBox_channel.setMaximum(self.nwa.NCHANNELS)
            self.spinBox_channel.valueChanged.connect(self.listWidget_traces.clear)
        else:
            self.spinBox_channel.setEnabled(False)

    def measure_traces(self):
        pass

    def measure_snp_network(self):
        pass

    def setup_nports(self):
        """
        nport data can be grabbed from the analyzer.  First we must specify the number of ports, then we must
        map the ports to the output measurement --> S11 = port1, S22 = port2... etc.
        :type nports: int
        :type mappable: boolean
        :return:
        """
        nports = self.comboBox_nports.currentIndex() + 1

        self.verticalLayout_snp.removeItem(self.spacerItem)
        self.verticalLayout_snp.removeWidget(self.btn_measureSnp)
        for row in self.snp_rows:
            self.verticalLayout_snp.removeItem(row)
            sip.delete(row.label)
            sip.delete(row.spinBox)
            sip.delete(row)

        self.snp_rows = []

        for port in range(1, nports + 1):
            layout = QtWidgets.QHBoxLayout()
            layout.label = QtWidgets.QLabel("Port{:d}".format(port))
            layout.spinBox = QtWidgets.QSpinBox()
            layout.spinBox.setValue(port)
            layout.spinBox.setMinimum(1)
            layout.spinBox.setMaximum(self.nwa.NPORTS)
            layout.addWidget(layout.label)
            layout.addWidget(layout.spinBox)
            self.verticalLayout_snp.addLayout(layout)
            self.snp_rows.append(layout)

        self.verticalLayout_snp.addWidget(self.btn_measureSnp)
        self.verticalLayout_snp.addItem(self.spacerItem)

    def update_traces(self):
        traces = self.nwa.get_list_of_traces()
        self.listWidget_traces.clear()
        for trace in traces:
            item = QtWidgets.QListWidgetItem()
            item.setText(trace["text"])
            self.listWidget_traces.addItem(item)
