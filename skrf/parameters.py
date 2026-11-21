"""
.. module:: skrf.parameters

========================================
parameters (:mod:`skrf.parameters`)
========================================

The arrays that a :class:`~skrf.network.Network` returns for its network
parameters, like ``ntwk.s`` or ``ntwk.z``, and for their scalar components,
like ``ntwk.s.db`` or ``ntwk.z.im``.

Both are :class:`numpy.ndarray` subclasses, so they behave like the arrays
they replace: they index, broadcast and go into numpy and scipy functions
as before, and ``ntwk.s`` is a view of the stored s-matrix, so writing into
it changes the Network. On top of that they know which Network they came
from, so they can plot themselves:

>>> ntwk.s.plot()             # Smith chart of every s-parameter
>>> ntwk.s.db.plot(m=1, n=0)  # |S21| in dB vs frequency
>>> ntwk.z.im[:, 0, 0]        # a plain numpy array

Anything numpy derives from them, a slice, a sum or ``ntwk.s * 2``, is a
plain :class:`numpy.ndarray` again.

Classes
-------

Only what these add to :class:`numpy.ndarray` is listed here.

.. autoclass:: NetworkParameter
    :members: plot, plot_smith, plot_complex, plot_polar

.. autoclass:: Projection
    :members: plot

"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from .constants import ComponentFuncT, PrimaryPropertiesT
    from .network import Network


class _NetworkArray(np.ndarray):
    """
    An array that knows the Network it belongs to.

    Only the array a Network hands out is tied to it. Everything numpy
    derives from it is a plain :class:`numpy.ndarray`, and it prints and
    pickles as one, so it can stand in wherever an array was expected.
    """

    _network: Network | None
    _name: PrimaryPropertiesT | None

    @classmethod
    def _attach(cls, values: np.ndarray, network: Network, name: PrimaryPropertiesT) -> Any:
        out = np.asarray(values).view(cls)
        out._network = network
        out._name = name
        return out

    def __array_finalize__(self, obj: Any) -> None:
        # views numpy makes of this array, like .T or .real, do not keep the
        # Network: they no longer hold the values it would plot
        self._network = None
        self._name = None

    def __array_wrap__(self, obj: np.ndarray, context: Any = None, return_scalar: bool = False) -> Any:
        # results of ufuncs and reductions are plain arrays
        out = obj.view(np.ndarray)
        return out[()] if return_scalar else out

    def __getitem__(self, key: Any) -> Any:
        out = super().__getitem__(key)
        return out.view(np.ndarray) if isinstance(out, np.ndarray) else out

    def __repr__(self) -> str:
        return repr(self.view(np.ndarray))

    def __reduce_ex__(self, protocol: Any) -> Any:
        # pickle as a plain array, it does not carry the Network along
        return self.view(np.ndarray).__reduce_ex__(protocol)

    def __copy__(self) -> np.ndarray:
        return self.view(np.ndarray).copy()

    def __deepcopy__(self, memo: dict) -> np.ndarray:
        return self.view(np.ndarray).copy()

    def copy(self, order: str = 'C') -> np.ndarray:
        return self.view(np.ndarray).copy(order=order)

    def _require_network(self) -> Network:
        if self._network is None:
            raise ValueError(
                "This array is not attached to a Network, so it cannot be plotted. "
                "Plot the parameter of the Network instead, like ntwk.s.db.plot()")
        return self._network


class NetworkParameter(_NetworkArray):
    """
    A network parameter matrix of a :class:`~skrf.network.Network`.

    This is what ``ntwk.s``, ``ntwk.z``, ``ntwk.y``, ``ntwk.t``, ``ntwk.a``,
    ``ntwk.g`` and ``ntwk.h`` return: a complex :class:`numpy.ndarray` of
    shape `fxnxn` that also knows its Network. ``ntwk.s`` is a view of the
    stored s-matrix, so ``ntwk.s[:, 0, 0] = 0`` changes the Network. The
    other parameters are computed from it, so writing into them does not.

    Its scalar components are attributes, each a :class:`Projection` that
    can plot itself: ``re``, ``im``, ``mag``, ``db``, ``db10``, ``rad``,
    ``deg``, ``arcl``, ``rad_unwrap``, ``deg_unwrap``, ``arcl_unwrap``,
    ``vswr``, ``time``, ``time_db`` and ``time_mag``. These hold the same
    values as the flat attributes, ``ntwk.s.db`` equals ``ntwk.s_db``.

    Examples
    --------
    >>> ntwk.s.plot()               # Smith chart
    >>> ntwk.z.plot()               # z in the complex plane
    >>> ntwk.s.deg.plot(m=1, n=0)   # phase of S21 vs frequency
    >>> ntwk.s.db.max()             # numpy methods work as usual

    See Also
    --------
    Projection
    skrf.network.Network.s
    """

    def _project(self, conversion: ComponentFuncT, func: Callable) -> Projection:
        out = Projection._attach(func(self.view(np.ndarray)), self._network, self._name)
        out._conversion = conversion
        return out

    def plot(self, m: int | None = None, n: int | None = None, **kwargs) -> None:
        """
        Plot the parameter.

        The s-parameters are drawn on a Smith chart, like
        :meth:`~skrf.network.Network.plot_s_smith`, and the other
        parameters in the complex plane, like
        :meth:`~skrf.network.Network.plot_prop_complex`.

        Parameters
        ----------
        m : int, optional
            first index of the parameter matrix, if None will use all
        n : int, optional
            second index of the parameter matrix, if None will use all
        \\*\\*kwargs : keyword arguments
            passed to the plot function, like `ax`, `show_legend` or
            :func:`matplotlib.pyplot.plot` arguments
        """
        if self._name == 's':
            return self.plot_smith(m=m, n=n, **kwargs)
        return self.plot_complex(m=m, n=n, **kwargs)

    def plot_smith(self, m: int | None = None, n: int | None = None, **kwargs) -> None:
        """
        Plot the s-parameters on a Smith chart.

        Only s-parameters can be drawn on a Smith chart. The arguments are
        those of :meth:`~skrf.network.Network.plot_s_smith`.
        """
        network = self._require_network()
        if self._name != 's':
            raise ValueError(f"Only s-parameters can be drawn on a Smith chart, not {self._name}-parameters")
        return network.plot_s_smith(m=m, n=n, **kwargs)

    def plot_complex(self, m: int | None = None, n: int | None = None, **kwargs) -> None:
        """
        Plot the parameter in the complex plane.

        The arguments are those of :meth:`~skrf.network.Network.plot_prop_complex`.
        """
        return self._require_network().plot_prop_complex(self._name, m=m, n=n, **kwargs)

    def plot_polar(self, m: int | None = None, n: int | None = None, **kwargs) -> None:
        """
        Plot the parameter in polar coordinates.

        The arguments are those of :meth:`~skrf.network.Network.plot_prop_polar`.
        """
        return self._require_network().plot_prop_polar(self._name, m=m, n=n, **kwargs)


class Projection(_NetworkArray):
    """
    A scalar component of a network parameter, like ``ntwk.s.db``.

    A real :class:`numpy.ndarray` of shape `fxnxn` with the same values as
    the flat attribute, ``ntwk.s.db`` equals ``ntwk.s_db``, that can plot
    itself against frequency, or against time for the ``time`` components.

    Examples
    --------
    >>> ntwk.s.db.plot()                 # every |Smn| in dB
    >>> ntwk.s.db.plot(m=1, n=0)         # |S21| in dB, same as ntwk.plot_s_db(m=1, n=0)
    >>> ntwk.s.db[:, 1, 0]               # a plain numpy array

    See Also
    --------
    NetworkParameter
    skrf.network.Network.plot_attribute
    """

    _conversion: ComponentFuncT | None

    def __array_finalize__(self, obj: Any) -> None:
        super().__array_finalize__(obj)
        self._conversion = None

    def plot(self, m: int | None = None, n: int | None = None, **kwargs) -> None:
        """
        Plot the component vs frequency, or vs time for the time components.

        Parameters
        ----------
        m : int, optional
            first index of the parameter matrix, if None will use all
        n : int, optional
            second index of the parameter matrix, if None will use all
        \\*\\*kwargs : keyword arguments
            passed to :meth:`~skrf.network.Network.plot_attribute`, like
            `ax`, `show_legend`, `y_label`, `logx` or
            :func:`matplotlib.pyplot.plot` arguments
        """
        network = self._require_network()
        return network.plot_attribute(self._name, self._conversion, m=m, n=n, **kwargs)
