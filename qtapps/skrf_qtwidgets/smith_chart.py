from math import sqrt, acos, degrees
import pyqtgraph as pg

R_vals_default = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 10, 20, 50]
X_vals_default = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 10, 20, 50]
ZGrid_pen = pg.mkPen("999999", width=0.2, antialias=True)
YGrid_pen = pg.mkPen("g", width=0.2, antialias=True)


def reactance_arc_sweep(X):
    """
    return the .arcTo parameters
    :param X: constant reactance
    :return: angle of the arc sweep
    """
    S11re = (X**2 - 1) / (X**2 + 1)  # real part of S11, when |S11| = 1
    S11im = sqrt(1 - S11re**2)  # imag part of S11, when |S11| = 1
    r = 1 / X
    c = sqrt((1 - S11re) ** 2 + S11im ** 2)
    theta = acos((2 * r ** 2 - c ** 2) / (2 * r ** 2))
    return degrees(theta)


def resistance_grid_lines(R_vals=R_vals_default, path_item=None):
    if path_item is None:
        path_item = pg.QtGui.QGraphicsPathItem()
    path = path_item.path()  # type: pg.QtGui.QPainterPath

    path.moveTo(1, 0)
    for R in R_vals:
        radius = 1 / (1 + R)
        path.addEllipse(1, -radius, -radius * 2, radius * 2)
    path_item.setPath(path)
    path_item.setPen(ZGrid_pen)

    return path_item


def conductance_grid_lines(C_vals=R_vals_default, path_item=None):
    if path_item is None:
        path_item = pg.QtGui.QGraphicsPathItem()
    path = path_item.path()  # type: pg.QtGui.QPainterPath

    path.moveTo(-1, 0)
    for C in C_vals:
        radius = 1 / (1 + C)
        path.addEllipse(-1, -radius, radius * 2, radius * 2)
    path_item.setPath(path)
    path_item.setPen(YGrid_pen)

    return path_item


def reactance_grid_lines(X_vals=X_vals_default, path_item=None):
    if path_item is None:
        path_item = pg.QtGui.QGraphicsPathItem()
    path = path_item.path()  # type: pg.QtGui.QPainterPath

    for X in X_vals:
        r = 1 / X
        d = 2 * r
        alpha = reactance_arc_sweep(X)
        x = 1 - r
        y = -d
        path.arcMoveTo(x, 0, d, d, 90)
        path.arcTo(x, 0, d, d, 90, alpha)
        path.arcMoveTo(x, y, d, d, -alpha - 90)
        path.arcTo(x, y, d, d, -alpha - 90, alpha)
    path.moveTo(1, 0)
    path.lineTo(-1, 0)
    path_item.setPath(path)
    path_item.setPen(ZGrid_pen)

    return path_item


def susceptance_grid_lines(Y_vals=X_vals_default, path_item=None):
    if path_item is None:
        path_item = pg.QtGui.QGraphicsPathItem()
    path = path_item.path()  # type: pg.QtGui.QPainterPath

    for Y in Y_vals:
        r = 1 / Y
        d = 2 * r
        alpha = reactance_arc_sweep(Y)
        x = -1 - r
        y = -d
        path.arcMoveTo(x, 0, d, d, -270)
        path.arcTo(x, 0, d, d, -270, -alpha)
        path.arcMoveTo(x, y, d, d, -90)
        path.arcTo(x, y, d, d, -90, alpha)

    path.moveTo(1, 0)
    path.lineTo(-1, 0)

    path_item.setPath(path)
    path_item.setPen(YGrid_pen)

    return path_item


def gen_s_unity_circle():
    s_unity_circle = pg.QtGui.QGraphicsEllipseItem(1, -1, -2, 2)
    s_unity_circle.setPen(pg.mkPen('w', antialias=True))
    return s_unity_circle


def gen_z_grid(major_accented=True):
    grid_lines = reactance_grid_lines(path_item=resistance_grid_lines())

    if major_accented:
        ZGridBold = reactance_grid_lines([1], path_item=resistance_grid_lines([1]))
        path = ZGridBold.path()
        path.moveTo(1, 0)
        path.lineTo(-1, 0)
        ZGridBold.setPen(pg.mkPen('999999', width=0.6, antialias=True))
        ZGridBold.setParentItem(grid_lines)

    return grid_lines


def gen_y_grid(major_accented=True):
    grid_lines = susceptance_grid_lines(path_item=conductance_grid_lines())

    if major_accented:
        YGridBold = susceptance_grid_lines([1], path_item=conductance_grid_lines([1]))
        path = YGridBold.path()
        path.moveTo(1, 0)
        path.lineTo(-1, 0)
        YGridBold.setPen(pg.mkPen('g', width=0.6, antialias=True))
        YGridBold.setParentItem(grid_lines)

    return grid_lines


if __name__ == "__main__":
    plot = pg.plot()
    plot.setAspectLocked()
    plot.setXRange(-1, 1)
    plot.setYRange(-1, 1)
    plot.addItem(gen_s_unity_circle())
    plot.addItem(gen_z_grid())
    plot.addItem(gen_y_grid())

    import sys
    if sys.flags.interactive != 1 or not hasattr(pg.QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()
