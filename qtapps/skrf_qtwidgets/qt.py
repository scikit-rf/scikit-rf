from __future__ import print_function

import os
import time
import sys
import traceback
import platform

import sip

from . import cfg  # must import cfg before qtpy to parse qt-bindings
os.environ['QT_API'] = 'pyqt5'  # force prefer pyqt5, let qtpy handle pyqt4 or pyside only
from qtpy import QtCore, QtWidgets, QtGui


def setup_style(style=cfg.preferred_style):
    available_styles = QtWidgets.QStyleFactory.keys()
    if style:
        if "QT_STYLE_OVERRIDE" in os.environ.keys():
            os.environ.pop("QT_STYLE_OVERRIDE")

        if style in available_styles:
            QtWidgets.QApplication.setStyle(style)
        else:
            for s in cfg.preferred_styles:
                if s in available_styles:
                    QtWidgets.QApplication.setStyle(s)

    elif platform.system() != "Windows" and os.environ["QT_API"] == "pyqt5":
        if "QT_STYLE_OVERRIDE" in os.environ.keys():
            os.environ.pop("QT_STYLE_OVERRIDE")
        if len(available_styles) == 2:
            # available styles are Windows, and Fusion
            # qt5-style-plugins are not installed, take action:
            for s in cfg.preferred_styles:
                if s in available_styles:
                    QtWidgets.QApplication.setStyle(s)


if os.environ['QT_API'] in ("pyqt", "pyqt4"):
    QtWidgets.QFileDialog.getOpenFileName = QtWidgets.QFileDialog.getOpenFileNameAndFilter
    QtWidgets.QFileDialog.getOpenFileNames = QtWidgets.QFileDialog.getOpenFileNamesAndFilter
    QtWidgets.QFileDialog.getSaveFileName = QtWidgets.QFileDialog.getSaveFileNameAndFilter


def excepthook_(type, value, tback):
    """overrides the default exception hook so that errors will print the error to the command line
    rather than just exiting with code 1 and no other explanation"""
    sys.__excepthook__(type, value, tback)
sys.excepthook = excepthook_


def popup_excepthook(type, value, tback):
    WarningMsgBox(traceback.format_exception(type, value, tback), "Uncaught Exception").exec_()


def set_popup_exceptions():
    sys.excepthook = popup_excepthook


# possibly necessary if the application needs matplotlib, as did previous versions of skrf
def reconcile_with_matplotlib():
    try:
        import matplotlib
        if os.environ['QT_API'] == 'pyqt5':
            matplotlib.use("Qt5Agg")
        elif os.environ['QT_API'] in ("pyqt", "pyqt4", "pyside"):
            matplotlib.use("Qt4Agg")
    except ImportError:
        print("matplotlib not installed, continuing")


def instantiate_app(sys_argv=None):
    if type(sys_argv) is None:
        sys_argv = []

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys_argv)
    return app


class WarningMsgBox(QtWidgets.QDialog):
    def __init__(self, text, title="Warning", parent=None):
        super(WarningMsgBox, self).__init__(parent)
        self.resize(500, 400)
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setContentsMargins(0, 0, 0, -1)
        self.textBrowser = QtWidgets.QTextBrowser(self)
        self.verticalLayout.addWidget(self.textBrowser)
        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        self.verticalLayout.addWidget(self.buttonBox)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.setWindowTitle(title)

        if type(text) in (list, tuple):
            text = "\n".join(text)
        self.textBrowser.setText(text)


def error_popup(error):
    if not type(error) is str:
        etype, value, tb = sys.exc_info()
        error = "\n".join(traceback.format_exception(etype, value, tb))
    WarningMsgBox(error).exec_()


def warnMissingFeature():
    msg = "Coming soon..."
    QtWidgets.QMessageBox.warning(None, "Feature Missing", msg, QtWidgets.QMessageBox.Ok)


# TODO: make the following dialogs function the same for PySide, PyQt4, PyQt5, and Possibly PySide2
def getOpenFileName_Global(caption, filter, start_path=None, **kwargs):
    if start_path is None:
        start_path = cfg.last_path
    fname = str(QtWidgets.QFileDialog.getOpenFileName(None, caption, start_path, filter, **kwargs)[0])
    if fname in ("", None):
        return ""
    cfg.last_path = os.path.dirname(fname)
    return fname


def getOpenFileNames_Global(caption, filter, start_path=None, **kwargs):
    if start_path is None:
        start_path = cfg.last_path
    fnames = QtWidgets.QFileDialog.getOpenFileNames(None, caption, start_path, filter, **kwargs)[0]
    fnames = [str(fname) for fname in fnames]
    if fnames in ("", None, []):
        return []
    cfg.last_path = os.path.dirname(fnames[0])
    return fnames


def getSaveFileName_Global(caption, filter, start_path=None, **kwargs):
    if start_path is None:
        start_path = cfg.last_path
    fname = str(QtWidgets.QFileDialog.getSaveFileName(None, caption, start_path, filter, **kwargs)[0])
    if fname in ("", None):
        return ""
    cfg.last_path = os.path.dirname(fname)
    return fname


def getDirName_Global(caption=None, start_path=None, **kwargs):
    if start_path is None:
        start_path = cfg.last_path
    dirname = str(QtWidgets.QFileDialog.getExistingDirectory(None, caption, start_path, **kwargs))
    if dirname in ("", None):
        return ""
    cfg.last_path = dirname
    return dirname


def center_widget(widget):
    widget.move(QtWidgets.QApplication.desktop().screen().rect().center() - widget.rect().center())


def get_skrf_icon():
    icon_image = os.path.join(cfg.images_dir, "scikit-rf-logo.png")
    return QtGui.QIcon(icon_image)


def get_splash_screen():
    splash_image = os.path.join(cfg.images_dir, "powered_by_scikit-rf.png")
    splash_pix = QtGui.QPixmap(splash_image)
    splash = QtWidgets.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()

    return splash, time.time()


def close_splash_screen(widget, splash, start_time):
    center_widget(widget)
    # make sure something bad doesn't happen
    if time.time() - start_time < 0:
        start_time = time.time()

    min_splash_time = 1
    while time.time() - start_time < min_splash_time:
        QtWidgets.QApplication.instance().processEvents()
    splash.finish(widget)


def single_widget_application(widget_class, splash_screen=True):
    app = QtWidgets.QApplication(sys.argv)

    setup_style()
    set_popup_exceptions()

    if splash_screen:
        splash, start_time = get_splash_screen()

    form = widget_class()
    form.setWindowIcon(get_skrf_icon())
    form.show()

    if splash_screen:
        close_splash_screen(form, splash, start_time)

    sip.setdestroyonexit(False)  # prevent a crash on exit
    sys.exit(app.exec_())


if __name__ == "main":
    # run some tests
    pass
