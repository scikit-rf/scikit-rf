def is_numeric(val):
    try:
        float(val)
        return True
    except ValueError:
        return False
