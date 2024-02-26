"""

.. currentmodule:: skrf.instances
========================================
instances (:mod:`skrf.instances`)
========================================

This module contains commonly used  instantiated objects's.



Standard Waveguide Bands
----------------------------

:class:`~skrf.frequency.Frequency` Objects
++++++++++++++++++++++++++++++++++++++++++++
These are predefined :class:`~skrf.frequency.Frequency` objects
that correspond to standard waveguide bands. This information is taken
from the VDI Application Note 1002 [#]_ . IEEE designators are taken
from Spinner TD-00036 [#]_ .


=======================  ===============================================
Object Name              Description
=======================  ===============================================
f_wr10                   WR-10, 75-110 GHz
f_wr3                    WR-3, 220-325 GHz
f_wr2p2                  WR-2.2, 330-500 GHz
f_wr1p5                  WR-1.5, 500-750 GHz
f_wr1                    WR-1, 750-1100 GHz
...                      ...
=======================  ===============================================


:class:`~skrf.media.rectangularWaveguide.RectangularWaveguide`  Objects
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
These are predefined :class:`~skrf.media.rectangularWaveguide.RectangularWaveguide`
objects for  standard waveguide bands.

=======================  ===============================================
Object Name              Description
=======================  ===============================================
wr10                     WR-10, 75-110 GHz
wr3                      WR-3, 220-325 GHz
wr2p2                    WR-2.2, 330-500 GHz
wr1p5                    WR-1.5, 500-750 GHz
wr1                      WR-1, 750-1100 GHz
...                      ...
=======================  ===============================================



References
----------
.. [#] VDI Application Note:  VDI Waveguide Band Designations (VDI-1002) http://vadiodes.com/VDI/pdf/waveguidechart200908.pdf
.. [#] Spinner Technical Information: Cross Reference For Hollow Metallic Waveguides (TD-00036) https://www.spinner-group.com/images/download/technical_documents/SPINNER_TD00036.pdf
"""

from .constants import mil
from .frequency import Frequency
from .media import Freespace, RectangularWaveguide


class StaticInstances:
    @property
    def air(self) -> Freespace:
        return Freespace()

    @property
    def air50(self) -> Freespace:
        return Freespace(z0_override=50)

    @property
    def f_wr51(self) -> Frequency:
        return Frequency(15,22,1001, 'GHz')

    @property
    def f_wr42(self) -> Frequency:
        return Frequency(17.5,26.5,1001, 'GHz')

    @property
    def f_wr34(self) -> Frequency:
        return Frequency(22,33,1001, 'GHz')

    @property
    def f_wr28(self) -> Frequency:
        return Frequency(26.5,40,1001, 'GHz')

    @property
    def f_wr22p4(self) -> Frequency:
        return Frequency(33,50.5,1001, 'GHz')

    @property
    def f_wr18p8(self) -> Frequency:
        return Frequency(40,60,1001, 'GHz')

    @property
    def f_wr14p8(self) -> Frequency:
        return Frequency(50,75,1001, 'GHz')

    @property
    def f_wr12p2(self) -> Frequency:
        return Frequency(60,90,1001, 'GHz')

    @property
    def f_wr10(self) -> Frequency:
        return Frequency(75,110,1001, 'GHz')

    @property
    def f_wr8(self) -> Frequency:
        return Frequency(90,140,1001, 'GHz')

    @property
    def f_wr6p5(self) -> Frequency:
        return Frequency(110,170,1001, 'GHz')

    @property
    def f_wr5p1(self) -> Frequency:
        return Frequency(140,220,1001, 'GHz')

    @property
    def f_wr4p3(self) -> Frequency:
        return Frequency(170,260,1001, 'GHz')

    @property
    def f_wr3p4(self) -> Frequency:
        return Frequency(220,330,1001, 'GHz')

    @property
    def f_wr2p8(self) -> Frequency:
        return Frequency(260,400,1001, 'GHz')

    @property
    def f_wr2p2(self) -> Frequency:
        return Frequency(330,500,1001, 'GHz')

    @property
    def f_wr1p9(self) -> Frequency:
        return Frequency(400,600,1001, 'GHz')

    @property
    def f_wr1p5(self) -> Frequency:
        return Frequency(500,750,1001, 'GHz')

    @property
    def f_wr1p2(self) -> Frequency:
        return Frequency(600,900,1001, 'GHz')

    @property
    def f_wr1(self) -> Frequency:
        return Frequency(750,1100,1001, 'GHz')

    @property
    def f_wr0p8(self) -> Frequency:
        return Frequency(900,1400,1001, 'GHz')

    @property
    def f_wr0p65(self) -> Frequency:
        return Frequency(1100,1700,1001, 'GHz')

    @property
    def f_wr0p51(self) -> Frequency:
        return Frequency(1400,2200,1001, 'GHz')

    @property
    def f_wm106(self) -> Frequency:
        return Frequency(1700,2600,1001, 'ghz')

    @property
    def f_wm86(self) -> Frequency:
        return Frequency(2200,3300,1001, 'ghz')

    @property
    def f_wm1295(self) -> Frequency:
        return self.f_wr5p1

    @property
    def f_wm1092(self) -> Frequency:
        return self.f_wr4p3

    @property
    def f_wm864(self) -> Frequency:
        return self.f_wr3p4

    @property
    def f_wm710(self) -> Frequency:
        return self.f_wr2p8

    @property
    def f_wm570(self) -> Frequency:
        return self.f_wr2p2

    @property
    def f_wm470(self) -> Frequency:
        return self.f_wr1p9

    @property
    def f_wm380(self) -> Frequency:
        return self.f_wr1p5

    @property
    def f_wm310(self) -> Frequency:
        return self.f_wr1p2

    @property
    def f_wm250(self) -> Frequency:
        return self.f_wr1

    @property
    def f_wm200(self) -> Frequency:
        return self.f_wr0p8

    @property
    def f_wm164(self) -> Frequency:
        return self.f_wr0p65

    @property
    def f_wm130(self) -> Frequency:
        return self.f_wr0p51

    @property
    def wr51(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr51.copy(),a=510*mil,b=255*mil,z0_override=50)

    @property
    def wr42(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr42.copy(),a=420*mil,b=170*mil,z0_override=50)

    @property
    def wr34(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr34.copy(),a=340*mil,b=170*mil,z0_override=50)

    @property
    def wr28(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr28.copy(),a=280*mil,b=140*mil,z0_override=50)

    @property
    def wr22p4(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr22p4.copy(),a=224*mil,b=112*mil,z0_override=50)

    @property
    def wr18p8(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr18p8.copy(),a=188*mil,b=94*mil,z0_override=50)

    @property
    def wr14p8(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr14p8.copy(),a=148*mil,b=74*mil,z0_override=50)

    @property
    def wr12p2(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr12p2.copy(),a=122*mil,b=61*mil,z0_override=50)

    @property
    def wr10(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr10.copy(),a=100*mil,b=50*mil,z0_override=50)

    @property
    def wr8(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr8.copy(),a=80*mil,b=40*mil,z0_override=50)

    @property
    def wr6p5(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr6p5.copy(),a=65*mil,b=32.5*mil,z0_override=50)

    @property
    def wr5p1(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr5p1.copy(),a=51*mil,b=25.5*mil,z0_override=50)

    @property
    def wr4p3(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr4p3.copy(),a=43*mil,b=21.5*mil,z0_override=50)

    @property
    def wr3p4(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr3p4.copy(),a=34*mil,b=17*mil,z0_override=50)

    @property
    def wr2p8(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr2p8.copy(),a=28*mil,b=14*mil,z0_override=50)

    @property
    def wr2p2(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr2p2.copy(),a=22*mil,b=11*mil,z0_override=50)

    @property
    def wr1p9(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr1p9.copy(),a=19*mil,b=9.5*mil,z0_override=50)

    @property
    def wr1p5(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr1p5.copy(),a=15*mil,b=7.5*mil,z0_override=50)

    @property
    def wr1p2(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr1p2.copy(),a=12*mil,b=6*mil,z0_override=50)

    @property
    def wr1(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr1.copy(),a=10*mil,b=5*mil,z0_override=50)

    @property
    def wr0p8(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr0p8.copy(),a=8*mil,b=4*mil,z0_override=50)

    @property
    def wr0p65(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr0p65.copy(),a=6.5*mil,b=3.25*mil,z0_override=50)

    @property
    def wr0p51(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wr0p51.copy(),a=5.1*mil,b=2.55*mil,z0_override=50)

    @property
    def wm1295(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wm1295.copy(),a=1295e-6,b=647.5e-6,z0_override=50)

    @property
    def wm1092(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wm1092.copy(),a=1092e-6,b=546e-6,z0_override=50)

    @property
    def wm864(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wm864.copy(),a=864e-6,b=432e-6,z0_override=50)

    @property
    def wm710(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wm710.copy(),a=710e-6,b=355e-6,z0_override=50)

    @property
    def wm570(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wm570.copy(),a=570e-6,b=285e-6,z0_override=50)

    @property
    def wm470(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wm470.copy(),a=470e-6,b=235e-6,z0_override=50)

    @property
    def wm380(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wm380.copy(),a=380e-6,b=190e-6,z0_override=50)

    @property
    def wm310(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wm310.copy(),a=310e-6,b=155e-6,z0_override=50)

    @property
    def wm250(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wm250.copy(),a=250e-6,b=125e-6,z0_override=50)

    @property
    def wm200(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wm200.copy(),a=200e-6,b=100e-6,z0_override=50)

    @property
    def wm164(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wm164.copy(),a=164e-6,b=82e-6,z0_override=50)

    @property
    def wm130(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wm130.copy(),a=130e-6,b=65e-6,z0_override=50)

    @property
    def wm106(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wm106.copy(),a=106e-6,b=53e-6,z0_override=50)

    @property
    def wm86(self) -> RectangularWaveguide:
        return RectangularWaveguide(self.f_wm86.copy(),a=86e-6,b=43e-6,z0_override=50)

_instances = StaticInstances()

def __getattr__(name):
    return getattr(_instances, name)
