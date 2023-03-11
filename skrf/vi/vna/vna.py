"""
.. module:: skrf.vi.vna.vna
=================================================
vna (:mod:`skrf.vi.vna.vna`)
=================================================

Provides the VNA base class
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

import functools
import inspect
import re
from abc import ABC
from enum import Enum, auto

import numpy as np
import pyvisa

from ..validators import Validator


def _format_cmd(cmd: str, **kwargs) -> str:
    def sub(match_obj):
        prefix = match_obj.group("prefix")
        attr = match_obj.group("attr")

        if prefix:
            obj = kwargs[prefix]
            return str(getattr(obj, attr))
        else:
            return str(kwargs[attr])

    param_re = re.compile(r"\<(?:(?P<prefix>\w+):)?(?P<attr>\w+)\>")
    return re.sub(param_re, sub, cmd)


class ValuesFormat(Enum):
    """How values are written to and queried from the insturment"""
    
    #: 32 bits per value
    BINARY_32 = auto()
    #: 64 bits per value
    BINARY_64 = auto()
    #: Transferred as ASCII (e.g. representing numbers as strings)
    ASCII = auto()


class Channel:
    """
    A single channel of the instrument. 

    This is only for those instruments which support channels, and should be
    subclassed in those instrument classes.

    .. warning::
        This class should not be instantiated directly
    """
    def __init__(
        self, parent, cnum: Optional[int] = None, cname: Optional[str]= None
    ) -> None:
        self.parent = parent
        self.cnum = cnum
        self.name = cname

        self.read = self.parent.read
        self.read_values = self.parent.read_values
        self.write = self.parent.write
        self.write_values = self.parent.write_values
        self.query = self.parent.query
        self.query_values = self.parent.query_values


class VNA(ABC):
    _scpi = True  # Set to false in subclasses that don't use SCPI

    def __init__(self, address: str, backend: str = "@py") -> None:
        rm = pyvisa.ResourceManager(backend)
        self._resource = rm.open_resource(address)

        if self._scpi:
            self._setup_scpi()

    def __init_subclass__(cls):
        if "Channel" in [c[0] for c in inspect.getmembers(cls, inspect.isclass)]:
            cls._add_channel_support()

    @classmethod
    def _add_channel_support(cls):
        def create_channel(self, cnum: int, cname: str) -> None:
            ch_id = f"ch{cnum}"
            if hasattr(self, ch_id):
                raise RuntimeError(f"Channel {cnum} already exists")

            setattr(self, ch_id, self.Channel(self, cnum, cname))

        def delete_channel(self, cnum: str) -> None:
            ch_id = f"ch{cnum}"
            if not hasattr(self, ch_id):
                return
            delattr(self, ch_id)

        def _channels(self) -> list[Channel]:
            return [
                getattr(self, ch) 
                for ch in dir(self) 
                if ch.startswith('ch') 
                and ch != 'channels'
            ]

        setattr(cls, "create_channel", create_channel)
        setattr(cls, "delete_channel", delete_channel)
        setattr(cls, "channels", property(_channels))
        # setattr(cls, "channels", property(lambda self: self.channels()))

    def _setup_scpi(self) -> None:
        setattr(
            self.__class__,
            "wait_for_complete",
            lambda self: self.query("*OPC?"),
        )
        setattr(self.__class__, "status", property(lambda self: self.query("*STB?")))
        setattr(self.__class__, "options", property(lambda self: self.query("*OPT?")))
        setattr(self.__class__, "id", property(lambda self: self.query("*IDN?")))

        # Reading and setting the query values format is instrument specific
        # and must be done for each subclass. We default to using ASCII
        self._values_fmt: ValuesFormat = ValuesFormat.ASCII

    @staticmethod
    def command(
        get_cmd: Optional[str] = None,
        set_cmd: Optional[str] = None,
        doc: Optional[str] = None,
        validator: Optional[Validator] = None,
        values: bool = False,
        values_container: Optional[type] = np.array,
    ) -> property:
        """
        Create a property for the instrument.

        This method is used to add a property to an instrument. These properties
        can be read-only, write-only, read-write, and can validate values before
        sending to the instrument as well as validate responses from the
        instrument to return proper types.

        Parameters
        ----------
        get_cmd
            Command sent to the instrument to request data
        set_cmd
            Command sent to the instrument to set data
        doc
            The docstring for the property
        validator
            The :class:`Validator` that will be used to transform data to the
            proper format before sending and after querying
        values
            Whether or not this command is using a `Sequence` to set data, or
            expects a `Sequence` in response.
        values_container:
            If values is true, you set set this to the type of container the
            values should be returned in. For example, this is np.array by
            default, meaning instead of return a `list`, you will get a numpy
            array.
            
        Returns
        -------
        property
            The property constructed from the parameters passed. Should be set
            to a class variable
        """

        def fget(self, get_cmd=get_cmd, validator=validator):
            if get_cmd is None:
                raise LookupError("Property cannot be read")

            cmd = _format_cmd(get_cmd, self=self)
            if values:
                arg = self.query_values(cmd, container=values_container)
            else:
                arg = self.query(cmd)

            if hasattr(self, 'wait_for_complete'):
                self.wait_for_complete()

            if validator:
                return validator.validate_output(arg)
            else:
                return arg

        def fset(self, arg, set_cmd=set_cmd, validator=validator):
            if set_cmd is None:
                raise LookupError("Property cannot be set")

            if validator:
                arg = validator.validate_input(arg)

            cmd = _format_cmd(set_cmd, self=self, arg=arg)
            self.write(cmd)

            if hasattr(self, 'wait_for_complete'):
                self.wait_for_complete()

        fget.__doc__ = doc
        # TODO: Potentially add the validator docstring to add the verbosity to
        # the generated docs, but keep the code less verbose?

        return property(fget=fget, fset=fset)

    def read(self, **kwargs) -> None:
        if isinstance(self._resource, pyvisa.resources.MessageBasedResource):
            fn = self._resource.read
        elif isinstance(self._resource, pyvisa.resources.RegisterBasedResource):
            raise NotImplementedError()
        else:
            raise RuntimeError("unreachable")

        return fn(**kwargs)

    def read_values(self, **kwargs) -> None:
        pass

    def write(self, cmd, **kwargs) -> None:
        if isinstance(self._resource, pyvisa.resources.MessageBasedResource):
            fn = self._resource.write
        elif isinstance(self._resource, pyvisa.resources.RegisterBasedResource):
            raise NotImplementedError()
        else:
            raise RuntimeError("unreachable")

        fn(cmd, **kwargs)

    def write_values(self, cmd, values, **kwargs) -> None:
        if isinstance(self._resource, pyvisa.resources.MessageBasedResource):
            if self._values_fmt == ValuesFormat.ASCII:
                fn = self._resource.write_ascii_values
            elif self._values_fmt == ValuesFormat.BINARY_32:
                fn = self._resource.write_binary_values
            elif self._values_fmt == ValuesFormat.BINARY_64:
                fn = functools.partial(self._resource.write_binary_values, datatype='d')
            
        elif isinstance(self._resource, pyvisa.resources.RegisterBasedResource):
            raise NotImplementedError()
        else:
            raise RuntimeError("unreachable")

        return fn(cmd, values, **kwargs)

    def query(self, cmd, **kwargs) -> None:
        if isinstance(self._resource, pyvisa.resources.MessageBasedResource):
            fn = self._resource.query
        elif isinstance(self._resource, pyvisa.resources.RegisterBasedResource):
            raise NotImplementedError()
        else:
            raise RuntimeError("unreachable")

        return fn(cmd, **kwargs)

    def query_values(self, cmd, **kwargs) -> None:
        if isinstance(self._resource, pyvisa.resources.MessageBasedResource):
            if self._values_fmt == ValuesFormat.ASCII:
                fn = self._resource.query_ascii_values
            elif self._values_fmt == ValuesFormat.BINARY_32:
                fn = self._resource.query_binary_values
            elif self._values_fmt == ValuesFormat.BINARY_64:
                fn = functools.partial(self._resource.query_binary_values, datatype='d')
        elif isinstance(self._resource, pyvisa.resources.RegisterBasedResource):
            raise NotImplementedError()
        else:
            raise RuntimeError("unreachable")

        return fn(cmd, **kwargs)
