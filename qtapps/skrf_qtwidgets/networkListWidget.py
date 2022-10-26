import os
import re
from collections import OrderedDict

from qtpy import QtWidgets, QtCore

import skrf
from . import qt, numeric_inputs, widgets
from .networkPlotWidget import NetworkPlotWidget


class NetworkListItem(QtWidgets.QListWidgetItem):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ntwk = None
        self.ntwk_corrected = None
        self.parameters = {}

    def update_ntwk_names(self, name):
        if isinstance(self.ntwk, skrf.Network):
            self.ntwk.name = name
        if isinstance(self.ntwk_corrected, skrf.Network):
            self.ntwk_corrected.name = name + "-cal"

    def set_text(self):
        self.setText(self.ntwk.name)


class NetworkListWidget(QtWidgets.QListWidget):
    item_removed = QtCore.Signal()
    item_updated = QtCore.Signal(object)
    save_single_requested = QtCore.Signal(object, str)
    selection_changed = QtCore.Signal(object)
    same_item_clicked = QtCore.Signal(object)
    state_changed = QtCore.Signal()

    def __init__(self, name_prefix='meas', parent=None):
        super().__init__(parent)
        self.name_prefix = name_prefix
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.customContextMenuRequested.connect(self.list_item_right_clicked)
        self.save_single_requested.connect(self.save_NetworkListItem)
        self.itemDelegate().commitData.connect(self.item_text_updated)
        self.item_updated.connect(self.set_active_network)
        self.item_updated.connect(self.state_changed.emit)
        self.item_removed.connect(self.state_changed.emit)

        self._ntwk_plot = None
        self.named_items = {}

        self.selected_items = set()
        self.selection_changed.connect(self.set_active_networks)
        self.same_item_clicked.connect(self.set_active_networks)

    def something_happened(self):
        # Create a set of the newly selected items, so we can
        # compare to the old selected items set
        newly_selected_items = set([item.text() for item in self.selectedItems()])
        if newly_selected_items != self.selected_items:
            # Only emit selection_changed signal if a change was detected
            self.selected_items = newly_selected_items
            self.selection_changed.emit(self.selected_items)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        newly_selected_items = set([item.text() for item in self.selectedItems()])
        if len(newly_selected_items) == 1 and newly_selected_items == self.selected_items:
            self.same_item_clicked.emit(self.selected_items)
        else:
            self.something_happened()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.something_happened()
    
    @staticmethod
    def save_NetworkListItem(ntwk_list_item, save_which):
        """
        Save a single network list item as a touchstone file
        
        Parameters
        ----------
        ntwk_list_item : NetworkListItem
            the QListWidgetItem referencing the N
        save_which : str
            specify to save the raw, corrected or both
        
        """

        if save_which.lower() not in ("raw", "cal", "both"):
            raise ValueError("Must set save option to 'raw', 'cal', or 'both'")

        ntwk = ntwk_list_item.ntwk
        ntwk_c = ntwk_list_item.ntwk_corrected

        if type(ntwk) in (list, tuple):
            widgets.save_multiple_networks(ntwk)
            return
        elif not isinstance(ntwk, skrf.Network):
            raise TypeError("ntwk must be a skrf.Network object to save")

        extension = f".s{ntwk.s.shape[1]:d}p"
        file_filter = f"touchstone format (*{extension:s})"
        filename = os.path.join(qt.cfg.last_path, ntwk.name + extension)

        if save_which.lower() == "both":
            if isinstance(ntwk, skrf.Network) and isinstance(ntwk_c, skrf.Network):
                filename = qt.getSaveFileName_Global("Save Raw skrf.Network File", filter=file_filter,
                                                     start_path=filename)
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
            names.append(item.ntwk.name)

        if name in names:
            if re.match(r"_\d\d", name[-3:]):
                name_base = name[:-3]
                suffix = int(name[-2:])
            else:
                name_base = name
                suffix = 1

            for num in range(suffix, 100, 1):
                name = f"{name_base:s}_{num:02d}"
                if name not in names:
                    break
        return name

    def item_text_updated(self, emit=True):
        item = self.currentItem()  # type: NetworkListItem
        item.setText(self.get_unique_name(item.text(), self.row(item)))
        item.update_ntwk_names(item.text())
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

            rename = QtWidgets.QAction("Rename Item", self)
            menu.addAction(rename)
            rename.triggered.connect(self.rename_item)

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

    def rename_item(self):
        item = self.currentItem()
        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        self.editItem(item)

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
        self.blockSignals(True)
        if item is None:
            return

        if self.ntwk_plot:
            self.ntwk_plot.set_networks(item.ntwk, item.ntwk_corrected)
        self.blockSignals(False)

    def load_named_ntwk(self, ntwk, name, activate=True):
        item = self.get_named_item(name)
        if not item:
            item = NetworkListItem()
            self.addItem(item)

        item.setFlags(item.flags() | QtCore.Qt.NoItemFlags)
        item.setText(name)
        item.ntwk = ntwk
        item.update_ntwk_names(name)

        self.clearSelection()
        self.setCurrentItem(item)
        if activate is True:
            self.set_active_network(item)

    def get_list_of_names(self):
        names = list()
        for i in range(self.count()):
            item = self.item(i)
            names.append(item.ntwk.name)
        return names

    def correction(self, ntwk):
        """
        method to for applying a correction to a network.  This method must be overridden to be functional

        Parameters
        ----------
        ntwk : skrf.Network
            the raw uncorrected network object

        Returns
        -------
        skrf.Network
            the corrected network object
        """
        return None

    def get_named_item(self, name):
        named_item = None
        for i in range(self.count()):
            item = self.item(i)
            if item.ntwk.name == name:
                named_item = item
        return named_item

    def load_network(self, ntwk, activate=True):
        item = NetworkListItem()
        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        name = self.get_unique_name(ntwk.name)
        item.ntwk = ntwk
        item.ntwk_corrected = self.correction(ntwk)
        item.update_ntwk_names(name)
        item.set_text()
        self.addItem(item)
        self.clearSelection()
        self.setCurrentItem(item)
        if activate:
            self.item_updated.emit(item)
        return item

    def load_from_file(self, caption="load touchstone file"):
        ntwk = widgets.load_network_file(caption)  # type: skrf.Network
        if not ntwk:
            return
        self.load_network(ntwk)

    def load_networks(self, ntwks):
        if not ntwks:
            return

        if isinstance(ntwks, skrf.Network):
            self.load_network(ntwks)
            return

        try:
            self.blockSignals(False)
            for ntwk in ntwks:
                self.load_network(ntwk, False)
        finally:
            self.blockSignals(False)

        item = self.item(self.count() - 1)
        self.set_active_network(item)

    def load_from_files(self, caption="load touchstone file"):
        self.load_networks(widgets.load_network_files(caption))

    def load_from_files_twoport(self, caption="load touchstone files"):
        self.load_networks(widgets.load_network_files(caption, filter="touchstone file (*.s2p)"))

    def save_selected_items(self):
        items = self.selectedItems()
        if len(items) == 1:
            item = items[0]
            self.save_single_requested.emit(item, self.get_save_which_mode())
        elif len(items) > 1:
            ntwk_list = []
            for item in items:
                ntwk_list.append(item.ntwk)
            widgets.save_multiple_networks(ntwk_list)

    def save_all_measurements(self):
        save_which = self.get_save_which_mode()
        ntwk_list = []

        for i in range(self.count()):
            item = self.item(i)
            if save_which != "cal" and isinstance(item.ntwk, skrf.Network):
                ntwk_list.append(item.ntwk)
            if save_which != "raw" and isinstance(item.ntwk_corrected, skrf.Network):
                ntwk_list.append(item.ntwk_corrected)

        widgets.save_multiple_networks(ntwk_list)

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
            dialog = widgets.MeasurementDialog(nwa)
            dialog.measurements_available.connect(self.load_networks)
            dialog.exec_()

    def measure_twoport(self, **kwargs):
        # TODO: get rid of this structure with measurement parameters
        with self.get_analyzer() as nwa:
            params = nwa.params_twoport.copy()
            params.update(kwargs)  # override any of the measurement_parameters with arguments passed in here
            meas = nwa.get_twoport(**params)
            meas.name = self.name_prefix  # unique name processed in load_network
        self.load_network(meas)

    def get_load_button(self, label="Load"):
        button = QtWidgets.QPushButton(label)
        button.released.connect(self.load_from_files)
        return button

    def get_load_button_twoport(self, label="Load"):
        button = QtWidgets.QPushButton(label)
        button.released.connect(self.load_from_files_twoport)
        return button

    def get_measure_button(self, label="Measure"):
        button = QtWidgets.QPushButton(label)
        button.released.connect(self.measure_ntwk)
        return button

    def get_measure_button_twoport(self, label="Measure"):
        button = QtWidgets.QPushButton(label)
        button.released.connect(self.measure_twoport)
        return button

    def get_save_selected_button(self, label="Save Selected"):
        button = QtWidgets.QPushButton(label)
        button.clicked.connect(self.save_selected_items)
        return button

    def get_save_all_button(self, label="Save All"):
        button = QtWidgets.QPushButton(label)
        button.clicked.connect(self.save_all_measurements)
        return button

    def get_input_buttons(self, labels=("Load", "Measure"), button_types="general"):
        widget = QtWidgets.QWidget()
        horizontal_layout = QtWidgets.QHBoxLayout(widget)
        horizontal_layout.setContentsMargins(0, 2, 0, 2)
        if button_types == "twoport":
            load_button = self.get_load_button_twoport(labels[0])
            measurement_button = self.get_measure_button_twoport(labels[1])
        elif button_types == "general":
            load_button = self.get_load_button(labels[0])
            measurement_button = self.get_measure_button(labels[1])
        else:
            raise TypeError("unrecognized request type for input buttons")
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


class ParameterizedNetworkListWidget(NetworkListWidget):
    def __init__(self, name_prefix="meas", item_parameters=(), parent=None):
        """
        initialize a parameterized version of the NetworkListWidget

        Parameters
        ----------
        name_prefix : str
            typically "meas", but some text that will be the default name of new measurements
        item_parameters : Iterable
            a list of dictionaries that contain the attributes each item will have
        parent : QtWidgets.QWidget
            the parent widget of the NetworkListWidget
        """
        super().__init__(name_prefix, parent)
        self.itemDelegate().commitData.disconnect(self.item_text_updated)
        self._item_parameters = OrderedDict()
        self.itemDoubleClicked.connect(self.edit_item)
        self._label_parameters = list()
        for param in item_parameters:
            self.add_item_parameter(param)

    @property
    def label_parameters(self):
        return self._label_parameters

    @label_parameters.setter
    def label_parameters(self, parameters):
        if type(parameters) is str:
            if parameters in self.item_parameters.keys():
                self._label_parameters = [parameters]
            else:
                print("invalid parameter")
                return
        elif type(parameters) not in (list, tuple):
            Warning("must provide a list of strings that identify parameters to include in the label")
            return

        label_parameters = list()

        for param_name in parameters:
            if type(param_name) is not str:
                print("must provide strings only")
            elif param_name in self.item_parameters.keys() and param_name not in self._label_parameters:
                label_parameters.append(param_name)

        if not label_parameters:
            Warning("no valid parameters provided")
        else:
            self._label_parameters = label_parameters

    def set_item_text(self, item):
        name = item.ntwk.name
        for param in self.label_parameters:
            name += f" - {item.parameters[param]}"
            units = self.item_parameters[param]["units"]
            if units:
                name += f" {units}"
        item.setText(name)

    def load_network(self, ntwk, activate=True, parameters=None):
        item = NetworkListItem()
        item.setFlags(item.flags() | QtCore.Qt.NoItemFlags)
        item.ntwk = ntwk
        if type(parameters) is dict:
            for name, param in self.item_parameters.items():
                item.parameters[name] = parameters.get(name, param["default"])
        else:
            for name, param in self.item_parameters.items():
                item.parameters[name] = param["default"]
        ntwk_name = self.get_unique_name(ntwk.name)
        item.update_ntwk_names(ntwk_name)
        self.addItem(item)
        self.clearSelection()
        self.setCurrentItem(item)
        self.set_item_text(item)
        if activate:
            self.set_active_network(item)
        return item

    def edit_item(self, item):
        """
        pop up a dialog to edit the name and network parameters

        Parameters
        ----------
        item : NetworkListItem
        """
        dialog = widgets.NetworkParameterEditor(item, self.item_parameters)

        state_changed = False

        accepted = dialog.exec_()
        if accepted:
            for name, input in dialog.inputs.items():
                if name == "name":
                    continue

                if isinstance(input, numeric_inputs.NumericLineEdit):
                    value = input.get_value()
                elif isinstance(input, QtWidgets.QLineEdit):
                    value = input.text()
                elif isinstance(input, QtWidgets.QComboBox):
                    value = input.currentText()
                elif isinstance(input, QtWidgets.QDoubleSpinBox) or isinstance(input, QtWidgets.QSpinBox):
                    value = input.value()
                else:
                    raise TypeError(f"Unsupported Widget {input}")
                if item.parameters[name] != value:
                    item.parameters[name] = value
                    state_changed = True

            index = self.indexFromItem(item).row()
            name = self.get_unique_name(dialog.inputs["name"].text(), index)
            if item.ntwk.name != name:
                item.update_ntwk_names(name)
                state_changed = True
            self.set_item_text(item)

            if state_changed:
                self.state_changed.emit()

    @property
    def item_parameters(self):
        return self._item_parameters

    def add_item_parameter(self, attribute):
        name = attribute["name"]
        attr_type = attribute["type"]
        default = attribute["default"]
        units = attribute.get("units", None)
        combo_list = attribute.get("combo_list", None)

        self._item_parameters[name] = {
            "type": attr_type,
            "default": default,
            "units": units,
            "combo_list": combo_list
        }

    def get_parameter_from_all(self, parameter):
        """
        retrieve the value of a parameter from the all of the items, and return as a list

        Parameters
        ----------
        parameter : str
            the dict key of the parameter of interest

        Returns
        -------
        list
        """
        values = list()
        for i in range(self.count()):
            item = self.item(i)
            values.append(item.parameters[parameter])
        return values