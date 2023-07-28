from skrf.vi.vna.keysight import FieldFox


class Analyzer(FieldFox):
    DEFAULT_VISA_ADDRESS = "TCPIP0::192.168.1.50::5025::SOCKET"
    NAME = "Keysight N9918A"
    NPORTS = 2
    NCHANNELS = 1
    SCPI_VERSION_TESTED = 'A.02.06.00'
