from PyQt5.QtWidgets import QTreeView,QFileSystemModel,QApplication

class Main(QTreeView):
    def __init__(self):
        QTreeView.__init__(self)
        model = QFileSystemModel()
        model.setRootPath('C:\\')
        self.setModel(model)
        self.doubleClicked.connect(self.test)

    def test(self, signal):
        file_path=self.model().filePath(signal)
        print(file_path)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Main()
    w.show()
    sys.exit(app.exec_())