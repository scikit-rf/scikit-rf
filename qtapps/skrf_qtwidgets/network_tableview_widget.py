import operator
import sys
import re

from skrf_qtwidgets import qt, util
# from . import util

from qtpy import QtCore, QtWidgets
import math
import skrf

number_units = re.compile("([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?|\s*[a-zA-Z]+\s*$)")


def parse_number_with_units(number_string):
    """
    :type number_string: str
    :return: list
    """
    matches = [match.group(0) for match in number_units.finditer(number_string)]
    if len(matches) not in (1, 2):
        return None

    try:
        value = float(matches[0])
    except ValueError:
        Warning("number_string does not contain valid number")
        return None

    units = "" if len(matches) == 1 else matches[1].strip()

    return value, units


def excepthook_(type, value, tback):
    """overrides the default exception hook so that errors will print the error to the command line
    rather than just exiting with code 1 and no other explanation"""
    sys.__excepthook__(type, value, tback)
sys.excepthook = excepthook_


class MyWindow(QtWidgets.QWidget):
    def __init__(self, data_list, header, *args):
        QtWidgets.QWidget.__init__(self, *args)
        self.setGeometry(300, 200, 570, 450)
        self.setWindowTitle("Click on column title to sort")

        table_view = QtWidgets.QTableView()
        row_height = math.ceil(self.logicalDpiX() / 96 * 20)  # attempt at dpi dependent scaling, yikes
        table_view.verticalHeader().setDefaultSectionSize(row_height)
        table_view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table_view.setShowGrid(False)

        table_model = NetworkTableModel(self, data_list, header)
        table_view.setModel(table_model)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(table_view)
        self.setLayout(layout)


class ThruItem(object):
    def __init__(self, network=None, value=None):
        self.ntwk = network
        self._name = network.name
        self._length = 0.0  # length in meters
        self.row = self.__class__.length

    def __len__(self):
        return 2

    def __getitem__(self, key):
        if key == 0:
            return self.name
        elif key == 1:
            return self.length
        else:
            raise IndexError("key {:} out of bounds".format(key))

    def __setitem__(self, key, value):
        if key == 0:
            self.name = value
        elif key == 1:
            self.length = value
        else:
            raise IndexError("key {:} out of bounds".format(key))

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = self.ntwk.name = name

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, val):
        try:
            self._length = float(val)
        except ValueError:
            number_with_units = parse_number_with_units(val)
            if not number_with_units:
                Warning("length must be in meters or with a unit of length, value {:} not recognized".format(val))
                return None
            length, units = number_with_units
            try:
                self._length = util.convert_length(float(length), units)
            except KeyError:
                Warning("invalid units {:} provided".format(units))
                return


class NetworkTableModel(QtCore.QAbstractTableModel):
    def __init__(self, parent, items, header, *args):
        super(NetworkTableModel, self).__init__(parent, *args)
        self.items = items
        self.header = header

    def rowCount(self, parent):
        return len(self.items)

    def columnCount(self, parent):
        return len(self.items[0])

    def data(self, index, role):
        if not index.isValid():
            return None
        elif role != QtCore.Qt.DisplayRole:
            return None
        return self.items[index.row()][index.column()]

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self.header[col]
        return None

    def sort(self, col, order):
        """sort table by given column number col"""
        self.layoutAboutToBeChanged.emit()
        self.items = sorted(self.items,
                            key=operator.itemgetter(col))
        if order == QtCore.Qt.DescendingOrder:
            self.items.reverse()
        self.layoutChanged.emit()

    def setData(self, index, value, role):
        """
        :param index: QModelIndex
        :type index: QModelIndex
        :param value: data value
        :type role: QtCore.Qt.ItemDataRole
        :return:
        """
        self.items[index.row()][index.column()] = value
        self.dataChanged.emit(index, index)
        return True

    def flags(self, index):
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable

one_port = skrf.Network(r"C:\Coding\Python\scikit-rf\qtapps\skrf_qtwidgets\example_data\horn antenna.s1p")
two_port = skrf.Network(r"C:\Coding\Python\scikit-rf\qtapps\skrf_qtwidgets\example_data\ring slot array simulation.s2p")

table_header = ['Network', 'Length (m)']
networks = [
    ThruItem(one_port, 0.0),
    ThruItem(two_port, 0.0),
]
app = QtWidgets.QApplication([])
win = MyWindow(networks, table_header)
win.show()
app.exec_()