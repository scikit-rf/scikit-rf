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

from . frequency import Frequency
from . media import RectangularWaveguide, Freespace
from .constants import mil
from .util import staticproperty

class StaticInstances:
    @staticproperty
    def air() -> Freespace:
        return Freespace()

    @staticproperty
    def air50() -> Freespace:
        return Freespace(z0_override=50)

    @staticproperty
    def f_wr51() -> Frequency:
        return Frequency(15,22,1001, 'GHz')

    @staticproperty
    def f_wr42() -> Frequency:
        return Frequency(17.5,26.5,1001, 'GHz')

    @staticproperty
    def f_wr34() -> Frequency:
        return Frequency(22,33,1001, 'GHz')

    @staticproperty
    def f_wr28() -> Frequency:
        return Frequency(26.5,40,1001, 'GHz')

    @staticproperty
    def f_wr22p4() -> Frequency:
        return Frequency(33,50.5,1001, 'GHz')

    @staticproperty
    def f_wr18p8() -> Frequency:
        return Frequency(40,60,1001, 'GHz')

    @staticproperty
    def f_wr14p8() -> Frequency:
        return Frequency(50,75,1001, 'GHz')

    @staticproperty
    def f_wr12p2() -> Frequency:
        return Frequency(60,90,1001, 'GHz')

    @staticproperty
    def f_wr10() -> Frequency:
        return Frequency(75,110,1001, 'GHz')

    @staticproperty
    def f_wr8() -> Frequency:
        return Frequency(90,140,1001, 'GHz')

    @staticproperty
    def f_wr6p5() -> Frequency:
        return Frequency(110,170,1001, 'GHz')

    @staticproperty
    def f_wr5p1() -> Frequency:
        return Frequency(140,220,1001, 'GHz')

    @staticproperty
    def f_wr4p3() -> Frequency:
        return Frequency(170,260,1001, 'GHz')

    @staticproperty
    def f_wr3p4() -> Frequency:
        return Frequency(220,330,1001, 'GHz')

    @staticproperty
    def f_wr2p8() -> Frequency:
        return Frequency(260,400,1001, 'GHz')

    @staticproperty
    def f_wr2p2() -> Frequency:
        return Frequency(330,500,1001, 'GHz')

    @staticproperty
    def f_wr1p9() -> Frequency:
        return Frequency(400,600,1001, 'GHz')

    @staticproperty
    def f_wr1p5() -> Frequency:
        return Frequency(500,750,1001, 'GHz')

    @staticproperty
    def f_wr1p2() -> Frequency:
        return Frequency(600,900,1001, 'GHz')

    @staticproperty
    def f_wr1() -> Frequency:
        return Frequency(750,1100,1001, 'GHz')

    @staticproperty
    def f_wr0p8() -> Frequency:
        return Frequency(900,1400,1001, 'GHz')

    @staticproperty
    def f_wr0p65() -> Frequency:
        return Frequency(1100,1700,1001, 'GHz')

    @staticproperty
    def f_wr0p51() -> Frequency:
        return Frequency(1400,2200,1001, 'GHz')
    
    @staticproperty
    def f_wm106() -> Frequency:
        return Frequency(1700,2600,1001, 'ghz')

    @staticproperty
    def f_wm86() -> Frequency:
        return Frequency(2200,3300,1001, 'ghz')

    @staticproperty
    def f_wm1295() -> Frequency:
        return StaticInstances.f_wr5p1

    @staticproperty
    def f_wm1092() -> Frequency:
        return StaticInstances.f_wr4p3

    @staticproperty
    def f_wm864() -> Frequency:
        return StaticInstances.f_wr3p4

    @staticproperty
    def f_wm710() -> Frequency:
        return StaticInstances.f_wr2p8

    @staticproperty
    def f_wm570() -> Frequency:
        return StaticInstances.f_wr2p2

    @staticproperty
    def f_wm470() -> Frequency:
        return StaticInstances.f_wr1p9

    @staticproperty
    def f_wm380() -> Frequency:
        return StaticInstances.f_wr1p5

    @staticproperty
    def f_wm310() -> Frequency:
        return StaticInstances.f_wr1p2

    @staticproperty
    def f_wm250() -> Frequency:
        return StaticInstances.f_wr1

    @staticproperty
    def f_wm200() -> Frequency:
        return StaticInstances.f_wr0p8

    @staticproperty
    def f_wm164() -> Frequency:
        return StaticInstances.f_wr0p65

    @staticproperty
    def f_wm130() -> Frequency:
        return StaticInstances.f_wr0p51

    @staticproperty
    def wr51() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr51.copy(),a=510*mil,b=255*mil,z0_override=50)

    @staticproperty
    def wr42() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr42.copy(),a=420*mil,b=170*mil,z0_override=50)

    @staticproperty
    def wr34() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr34.copy(),a=340*mil,b=170*mil,z0_override=50)

    @staticproperty
    def wr28() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr28.copy(),a=280*mil,b=140*mil,z0_override=50)

    @staticproperty
    def wr22p4() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr22p4.copy(),a=224*mil,b=112*mil,z0_override=50)

    @staticproperty
    def wr18p8() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr18p8.copy(),a=188*mil,b=94*mil,z0_override=50)

    @staticproperty
    def wr14p8() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr14p8.copy(),a=148*mil,b=74*mil,z0_override=50)

    @staticproperty
    def wr12p2() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr12p2.copy(),a=122*mil,b=61*mil,z0_override=50)

    @staticproperty
    def wr10() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr10.copy(),a=100*mil,b=50*mil,z0_override=50)

    @staticproperty
    def wr8() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr8.copy(),a=80*mil,b=40*mil,z0_override=50)

    @staticproperty
    def wr6p5() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr6p5.copy(),a=65*mil,b=32.5*mil,z0_override=50)

    @staticproperty
    def wr5p1() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr5p1.copy(),a=51*mil,b=25.5*mil,z0_override=50)

    @staticproperty
    def wr4p3() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr4p3.copy(),a=43*mil,b=21.5*mil,z0_override=50)

    @staticproperty
    def wr3p4() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr3p4.copy(),a=34*mil,b=17*mil,z0_override=50)

    @staticproperty
    def wr2p8() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr2p8.copy(),a=28*mil,b=14*mil,z0_override=50)

    @staticproperty
    def wr2p2() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr2p2.copy(),a=22*mil,b=11*mil,z0_override=50)

    @staticproperty
    def wr1p9() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr1p9.copy(),a=19*mil,b=9.5*mil,z0_override=50)

    @staticproperty
    def wr1p5() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr1p5.copy(),a=15*mil,b=7.5*mil,z0_override=50)

    @staticproperty
    def wr1p2() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr1p2.copy(),a=12*mil,b=6*mil,z0_override=50)

    @staticproperty
    def wr1() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr1.copy(),a=10*mil,b=5*mil,z0_override=50)

    @staticproperty
    def wr0p8() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr0p8.copy(),a=8*mil,b=4*mil,z0_override=50)

    @staticproperty
    def wr0p65() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr0p65.copy(),a=6.5*mil,b=3.25*mil,z0_override=50)

    @staticproperty
    def wr0p51() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wr0p51.copy(),a=5.1*mil,b=2.55*mil,z0_override=50)

    @staticproperty
    def wm1295() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wm1295.copy(),a=1295e-6,b=647.5e-6,z0_override=50)

    @staticproperty
    def wm1092() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wm1092.copy(),a=1092e-6,b=546e-6,z0_override=50)

    @staticproperty
    def wm864() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wm864.copy(),a=864e-6,b=432e-6,z0_override=50)

    @staticproperty
    def wm710() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wm710.copy(),a=710e-6,b=355e-6,z0_override=50)

    @staticproperty
    def wm570() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wm570.copy(),a=570e-6,b=285e-6,z0_override=50)

    @staticproperty
    def wm470() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wm470.copy(),a=470e-6,b=235e-6,z0_override=50)

    @staticproperty
    def wm380() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wm380.copy(),a=380e-6,b=190e-6,z0_override=50)

    @staticproperty
    def wm310() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wm310.copy(),a=310e-6,b=155e-6,z0_override=50)

    @staticproperty
    def wm250() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wm250.copy(),a=250e-6,b=125e-6,z0_override=50)

    @staticproperty
    def wm200() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wm200.copy(),a=200e-6,b=100e-6,z0_override=50)

    @staticproperty
    def wm164() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wm164.copy(),a=164e-6,b=82e-6,z0_override=50)

    @staticproperty
    def wm130() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wm130.copy(),a=130e-6,b=65e-6,z0_override=50)

    @staticproperty
    def wm106() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wm106.copy(),a=106e-6,b=53e-6,z0_override=50)

    @staticproperty
    def wm86() -> RectangularWaveguide:
        return RectangularWaveguide(StaticInstances.f_wm86.copy(),a=86e-6,b=43e-6,z0_override=50)


def __getattr__(name):
    return getattr(StaticInstances, name)