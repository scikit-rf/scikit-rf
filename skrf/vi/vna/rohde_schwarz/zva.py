from .rs_vna import RSVNA


class ZVA(RSVNA):
    """
    Rohde & Schwarz ZVA.

    ZVA Models
    ==========
    ZVA40, ..., others

    """

    _models = {
        "default": {"nports": 2, "unsupported": []},
    }
