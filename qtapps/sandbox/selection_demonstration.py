from PyQt4 import QtGui
QtWidgets = QtGui
# from PyQt5 import QtWidgets
import sys
import time


def something_happened():
    print(time.time())

app = QtWidgets.QApplication(sys.argv)

widget = QtWidgets.QWidget(None)
vlay = QtWidgets.QVBoxLayout(widget)

list_widget1 = QtWidgets.QListWidget(None)
list_widget1.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
list_widget1.addItems(("Cheese", "Whiz", "tastes", "great"))
list_widget1.itemSelectionChanged.connect(something_happened)
list_widget1.clicked.connect(something_happened)

list_widget2 = QtWidgets.QListWidget(None)
list_widget2.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
list_widget2.addItems(("No", "it", "tastes", "bad"))
list_widget2.itemSelectionChanged.connect(something_happened)
# list_widget2.clicked.connect(something_happened)

vlay.addWidget(list_widget1)
vlay.addWidget(list_widget2)

widget.show()

app.exec_()
