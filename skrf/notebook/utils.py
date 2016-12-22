from collections import OrderedDict

colors = OrderedDict()
colors["lime_green"] = "#00FF00"
colors["green"] = "#00AA00"
colors["cyan"] = "#00FFFF"
colors["blue"] = "#0000FF"
colors["red"] = "#FF0000"
colors["magenta"] = "#FF00FF"
colors["yellow"] = "#FFFF00"
colors["purple"] = "#990099"
colors["orange"] = "#FFA500"


def trace_color_cycle(start=0):
    """
    :start n: int
    :return:
    """
    count = start
    color_list = [colors["blue"], colors["red"], colors["magenta"], colors["green"]]
    num = len(color_list)
    while count < 1000:
        yield color_list[count % num]
        count += 1