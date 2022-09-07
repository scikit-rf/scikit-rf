from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List

import pyvisa


def available(backend: str = "@py") -> List[str]:
    rm = pyvisa.ResourceManager(backend)
    avail = rm.list_resources()
    rm.close()
    return list(avail)
