import re

from qtpy import QtWidgets, QtCore, QtGui

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


class NumericLineEdit(QtWidgets.QLineEdit):
    def __init__(self, parent=None):
        super(NumericLineEdit, self).__init__(parent)

    def sizeHint(self):
        return QtCore.QSize(60, 22)

    def get_value(self):
        return float(self.text())


class DoubleLineEdit(NumericLineEdit):
    def __init__(self, value=0, parent=None):
        super(DoubleLineEdit, self).__init__(parent)
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
        super(InputWithUnits, self).__init__(parent)
        if value is not None:
            try:
                value = float(value)
                self.setText(str(value))
            except ValueError as e:
                Warning("invalid entry {:} for line Edit".format(value))

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
            self.setText("{:0.6g}".format(value))
        elif unit in self.conversions.keys():
            value *= self.conversions[self.units] / self.conversions[unit]
            self.setText("{:0.6g}".format(value))
        else:
            self.setText("invalid unit")

    def get_value(self, units=None):
        value = float(self.text())

        if type(units) is str:
            if units in self.conversions.keys():
                value *= self.conversions[self.units] / self.conversions[units]
            else:
                raise ValueError("Invalid units {:} provided to get_value".format(units))

        return value
