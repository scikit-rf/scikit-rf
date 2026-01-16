from .rs_vna import RSVNA


class ZNA(RSVNA):
    """
    Rohde & Schwarz ZNA.

    ZNA Models
    ==========
    ZNA26-2Port, ZNA26-4Port, ZNA43-2Port, ZNA43-4Port, ZNA50-2Port, ZNA50-4Port, ZNA67-2Port, ZNA67-4Port

    """

    _models = {
        "default": {"nports": 2, "unsupported": []},
        "ZNA26-2Port": {"nports": 2, "unsupported": []},
        "ZNA26-4Port": {"nports": 4, "unsupported": []},
        "ZNA43-2Port": {"nports": 2, "unsupported": []},
        "ZNA43-4Port": {"nports": 4, "unsupported": []},
        "ZNA50-2Port": {"nports": 2, "unsupported": []},
        "ZNA50-4Port": {"nports": 4, "unsupported": []},
        "ZNA67-2Port": {"nports": 2, "unsupported": []},
        "ZNA67-4Port": {"nports": 4, "unsupported": []},
    }
