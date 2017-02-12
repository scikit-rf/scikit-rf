class SomeFrame(object):
    """A container to place all the widgets, and control
       present the output from the selections.
    """

    def __init__(self):
        self.frame = QtWidgets.QFrame()

        # Creating content
        list_widget1 = MyQListWidgetC()
        list_widget1.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        list_widget1.addItems(("Cheese", "Whiz", "tastes", "great"))

        list_widget2 = MyQListWidgetC()
        list_widget2.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        list_widget2.addItems(("No", "it", "tastes", "bad"))

        # Creating Layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(list_widget1)
        layout.addWidget(list_widget2)

        self.frame.setLayout(layout)

        # Connections
        from functools import partial
        list_widget1.selection_changed.connect(partial(self.selectionChangedCB, 'list_1'))
        list_widget1.same_item_clicked.connect(partial(self.selectionChangedCB, 'list_1'))
        list_widget2.selection_changed.connect(partial(self.selectionChangedCB, 'list_2'))
        list_widget2.same_item_clicked.connect(partial(self.selectionChangedCB, 'list_2'))

        self.frame.show()

    def selectionChangedCB(self, list_name, selected_items):
        print(list_name + ' changed: ' + str(selected_items))


from PyQt5 import QtWidgets, QtCore


class MyQListWidgetA(QtWidgets.QListWidget):
    """This widget emits selection_changed whenever its
       itemSelectionChanged signal is emitted, AND there
       was an actual change in the selected items.
    """
    selection_changed = QtCore.pyqtSignal(object)

    def __init__(self):
        QtWidgets.QListWidget.__init__(self)

        self.selected_items = set()

        self.itemSelectionChanged.connect(self.something_happened)

    def something_happened(self):
        # Create a set of the newly selected items, so we can compare
        # to the old selected items set
        newly_selected_items = set([item.text() for item in self.selectedItems()])
        if newly_selected_items != self.selected_items:
            # Only emit selection_changed signal if a change was detected.
            self.selected_items = newly_selected_items
            self.selection_changed.emit(self.selected_items)


class MyQListWidgetB(QtWidgets.QListWidget):
    """This widget emits selection_changed whenever it is pressed
       (mimic "clicked" signal) and again when the user is done
       selecting the items (mouse release) IFF the selection
       has changed.
    """
    selection_changed = QtCore.pyqtSignal(object)

    def __init__(self):
        QtWidgets.QListWidget.__init__(self)

        self.selected_items = set()

    def something_happened(self, initial_click=False):
        # Create a set of the newly selected items, so we can
        # compare to the old selected items set
        newly_selected_items = set([item.text() for item in self.selectedItems()])
        if newly_selected_items != self.selected_items:
            # Only emit selection_changed signal if a change was detected
            self.selected_items = newly_selected_items
            self.selection_changed.emit(self.selected_items)

    def mousePressEvent(self, event):
        QtWidgets.QListWidget.mousePressEvent(self, event)
        self.something_happened()

    def mouseReleaseEvent(self, event):
        QtWidgets.QListWidget.mouseReleaseEvent(self, event)
        self.something_happened()

class MyQListWidgetC(QtWidgets.QListWidget):
    """This widget emits selection_changed whenever it is pressed
       (mimic "clicked" signal) and again when the user is done
       selecting the items (mouse release) IFF the selection
       has changed. If a single item was clicked, AND it is the
       same item, the widget will emit same_item_clicked, which
       the owner can listen to and decide what to do.
    """
    selection_changed = QtCore.pyqtSignal(object)
    same_item_clicked = QtCore.pyqtSignal(object)

    def __init__(self):
        QtWidgets.QListWidget.__init__(self)

        self.selected_items = set()

    def something_happened(self, initial_click=False):
        # Create a set of the newly selected items, so we can
        # compare to the old selected items set
        newly_selected_items = set([item.text() for item in self.selectedItems()])
        if newly_selected_items != self.selected_items:
            # Only emit selection_changed signal if a change was detected
            self.selected_items = newly_selected_items
            self.selection_changed.emit(self.selected_items)

    def mousePressEvent(self, event):
        QtWidgets.QListWidget.mousePressEvent(self, event)
        newly_selected_items = set([item.text() for item in self.selectedItems()])
        if len(newly_selected_items) == 1 and newly_selected_items == self.selected_items:
            self.same_item_clicked.emit(self.selected_items)
        else:
            self.something_happened()

    def mouseReleaseEvent(self, event):
        QtWidgets.QListWidget.mouseReleaseEvent(self, event)
        self.something_happened()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    theFrame = SomeFrame()
    sys.exit(app.exec_())