from skrf.vi.vna import rs_zva


class Analyzer(rs_zva.ZVA):
    DEFAULT_VISA_ADDRESS = "GPIB::16::INSTR"
    NAME = "Rhode & Schwartz ZVA"
    NPORTS = 4
    NCHANNELS = 32
    SCPI_VERSION_TESTED = ''
