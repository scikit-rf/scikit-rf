import operator
import sys

from qtpy import QtCore, QtWidgets
import math


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


class TableItem(object):
    def __init__(self, network=None, value=None):
        self.ntwk = network
        self.row = [self.ntwk, value]


class NetworkTableModel(QtCore.QAbstractTableModel):
    def __init__(self, parent, myarray, header, *args):
        super(NetworkTableModel, self).__init__(parent, *args)
        self.items = myarray
        self.header = header

    def rowCount(self, parent):
        return len(self.items)

    def columnCount(self, parent):
        return len(self.items[0].row)

    def data(self, index, role):
        if not index.isValid():
            return None
        elif role != QtCore.Qt.DisplayRole:
            return None
        return self.items[index.row()].row[index.column()]

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
        item = self.items[index.row()]
        item.setValue(index.column(), value)
        self.dataChanged.emit()
        return True

    def flags(self, index):
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable

header = ['Network', 'Length (mm)']
data_list = [
    TableItem("Reflect", 0.0),
    TableItem("Thru", 0.0),
    TableItem("Line", 10.0)
]
app = QtWidgets.QApplication([])
win = MyWindow(data_list, header)
win.show()
app.exec_()