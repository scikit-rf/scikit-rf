import os
import time
import sys
import traceback
import platform
import ctypes

import sip
from . import cfg  # must import cfg before qtpy to properly parse qt-bindings
from qtpy import QtCore, QtWidgets, QtGui


class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class QVLine(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


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


def reconcile_with_matplotlib():
    """this function makes sure that matplotlib is using the preferred version of Qt before we instantiate our
    python qt-bindings libraries"""
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
        super().__init__(parent)
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
    return QtGui.QIcon(cfg.skrf_icon)


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


def single_widget_application(widget_class, splash_screen=True, appid="skrf.qtapp", icon=cfg.skrf_icon):
    if appid:
        set_process_id(appid)

    app = QtWidgets.QApplication(sys.argv)

    setup_style()
    set_popup_exceptions()

    if splash_screen:
        splash, start_time = get_splash_screen()

    form = widget_class()
    try:
        if type(icon) is str:
            icon = QtGui.QIcon(icon)
        elif not isinstance(icon, QtGui.QIcon):
            icon = False
        if icon:
            form.setWindowIcon(icon)
    except Exception as e:
        error_popup(e)

    form.show()

    if splash_screen:
        close_splash_screen(form, splash, start_time)

    sip.setdestroyonexit(False)  # prevent a crash on exit
    sys.exit(app.exec_())


def set_process_id(appid=None):
    """
    in windows, setting this parameter allows all instances to be grouped under the same taskbar icon, and allows
    us to set an icon that is different from whatever the python executable is using.
    :param appid: str, unicode
    :return:
    """
    if appid and type(appid) is str and platform.system() == "Windows":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)


class HTMLDisplay(QtWidgets.QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.defaultSize = super().sizeHint()
        self.h = self.defaultSize.height() * 2
        self.w = self.defaultSize.width() * 2

    def sizeHint(self):
        size = QtCore.QSize()
        size.setHeight(self.h)
        size.setWidth(self.w)
        return size


class HelpIndicator(QtWidgets.QPushButton):
    def __init__(self, title="help text", help_text=None, parent=None):
        super().__init__(parent)
        self.defaultSize = super().sizeHint()
        self.title = title
        self.h = self.defaultSize.height()
        self.w = self.h
        self.setFixedWidth(self.w)
        self.setText("?")
        if help_text and type(help_text) is str:
            self.help_text = help_text
        else:
            self.help_text = None
        self.clicked.connect(self.popup)

    def sizeHint(self):
        size = QtCore.QSize()
        size.setHeight(self.h)
        size.setWidth(self.w)
        return size

    def popup(self):
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle(self.title)
        vlay = QtWidgets.QVBoxLayout(dialog)
        vlay.setContentsMargins(2, 2, 2, 2)
        textEdit = HTMLDisplay()
        qsize = textEdit.sizeHint()  # type: QtCore.QSize
        qsize.setWidth(qsize.width() * 2)
        qsize.setHeight(qsize.height() * 2)
        textEdit.setReadOnly(True)
        textEdit.setHtml(self.help_text)
        vlay.addWidget(textEdit)
        dialog.exec_()


class RunFunctionDialog(QtWidgets.QDialog):
    def __init__(self, function, title="Wait", text=None, parent=None):
        super().__init__(parent)
        self.function = function
        self.layout = QtWidgets.QVBoxLayout(self)
        if text is None:
            text = "Running Function, will close automatically"
        self.text = QtWidgets.QTextBrowser()
        self.text.setText(text)
        self.layout.addWidget(self.text)

    def showEvent(self, QShowEvent):
        self.function()
