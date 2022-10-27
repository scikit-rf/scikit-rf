import re

from qtpy import QtWidgets, QtCore, QtGui

from . import util

available_units = {
    "frequency": {
        "base": "Hz",
        "Hz": 1.0, "kHz": 1e-3, "MHz": 1e-6, "GHz": 1e-9, "THz": 1e-12, "PHz": 1e-15
    },
    "length": {
        "base": "m",
        "m": 1.0, "dm": 10, "cm": 100, "mm": 1e3, "um": 1e6, "nm": 1e9, "pm": 1e12, "km": 1e-3,
        "yd": 1000 / 25.4 / 36,  "ft": 1000 / 25.4 / 12, "in": 1000 / 25.4, "mil": 1e6 / 25.4
    }
}

number_units = re.compile(r"([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?|\s*[a-zA-Z]+\s*$)")


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


class NumericLineEdit(QtWidgets.QLineEdit):
    value_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_value = self.text()
        self.editingFinished.connect(self.check_state)

    def check_state(self):
        if not self.current_value == self.text():
            self.current_value = self.text()
            self.value_changed.emit()

    def sizeHint(self):
        return QtCore.QSize(60, 22)

    def get_value(self):
        return float(self.text())

    def set_value(self, value):
        if type(value) in (float, int):
            str_val = f"{value:0.6g}"
        elif util.is_numeric(value):
            str_val = str(value)
        else:
            raise TypeError("must provide a number or a numeric string")

        self.setText(str_val)
        self.check_state()


class DoubleLineEdit(NumericLineEdit):
    def __init__(self, value=0, parent=None):
        super().__init__(parent)
        self.setValidator(QtGui.QDoubleValidator())
        self.setText(str(value))


class InputWithUnits(NumericLineEdit):
    """define a QLineEdit box that parses a number with units, and converts to a default unit base"""

    def __init__(self, units, value=None, parent=None):
        """
        Parameters
        ----------
        units : str
            the unit of measure, e.g. "Hz", "mm", "in", "mil"
        """
        super().__init__(parent)
        self.editingFinished.disconnect(self.check_state)
        if value is not None:
            try:
                value = float(value)
                self.setText(str(value))
            except ValueError as e:
                Warning(f"invalid entry {value} for line Edit")

        self.units = units
        self.conversions = None
        for key, unit_list in available_units.items():
            if units in unit_list.keys():
                self.conversions = unit_list
                self.base_unit = self.conversions["base"]
        if not self.conversions:
            raise ValueError("unit not recognized")
        self.editingFinished.connect(self.number_entered)

    def number_entered(self):
        value, unit = parse_number_with_units(self.text())
        if not unit:
            self.setText(f"{value:0.6g}")
        elif unit in self.conversions.keys():
            value *= self.conversions[self.units] / self.conversions[unit]
            self.setText(f"{value:0.6g}")
        else:
            self.setText("invalid unit")
        self.check_state()

    def get_value(self, units=None):
        value = float(self.text())
        if type(units) is str:
            if units in self.conversions.keys():
                value *= self.conversions[units] / self.conversions[self.units]
            else:
                raise ValueError(f"Invalid units {units} provided to get_value")
        return value

    def set_units(self, units):
        if units not in self.conversions.keys():
            raise KeyError(f"invalid units {units}")

        value = float(self.text()) / self.conversions[self.units] * self.conversions[units]
        self.units = units
        self.setText(f"{value:0.6g}")
        self.check_state()
