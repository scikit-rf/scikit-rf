def debug_counter(n=-1):
    count = 0
    while count != n:
        count += 1
        yield count


def is_numeric(val):
    try:
        float(val)
        return True
    except ValueError:
        return False


def trace_color_cycle(n=1000):
    """
    :type n: int
    :return:
    """

    lime_green = "#00FF00"
    cyan = "#00FFFF"
    magenta = "#FF00FF"
    yellow = "#FFFF00"
    pink = "#C04040"
    blue = "#0000FF"
    lavendar = "#FF40FF"
    turquoise = "#00FFFF"

    count = 0
    colors = [yellow, cyan, magenta, lime_green, pink, blue, lavendar, turquoise]
    num = len(colors)
    while count < n:
        yield colors[count % num]
        count += 1
