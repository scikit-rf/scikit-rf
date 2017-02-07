length_units = {
    "m": 1.0,
    "dm": 0.1,
    "cm": 0.01,
    "mm": 0.001,
    "um": 1e-6,
    "nm": 1e-9,
    "in": 0.0254,
    "mil": 0.0254 / 1000,
    "ft": 0.0254 * 12,
}

def is_numeric(val):
    try:
        float(val)
        return True
    except ValueError:
        return False

def convert_length(length, from_units, to_units="m"):
    return length * length_units[from_units.lower()] / length_units[to_units.lower()]
