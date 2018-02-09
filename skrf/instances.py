

'''

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
from Spinner TD-00036 [*]_ .


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
-------------
.. [#] VDI Application Note:  VDI Waveguide Band Designations (VDI-1002) http://vadiodes.com/VDI/pdf/waveguidechart200908.pdf
.. [*] Spinner Technical Information: Cross Reference For Hollow Metallic Waveguides (TD-00036) https://www.spinner-group.com/images/download/technical_documents/SPINNER_TD00036.pdf
'''

from . frequency import Frequency
from . media import RectangularWaveguide, Freespace, DefinedGammaZ0
from .constants import mil




air = Freespace()
air50 = Freespace(z0=50)


######## waveguide bands
f_wr51  = Frequency(15,22,1001, 'ghz')
f_wr42  = Frequency(17.5,26.5,1001, 'ghz')
f_wr34  = Frequency(22,33,1001, 'ghz')
f_wr28  = Frequency(26.5,40,1001, 'ghz')
f_wr22p4  = Frequency(33,50.5,1001, 'ghz')
f_wr18p8  = Frequency(40,60,1001, 'ghz')
f_wr14p8  = Frequency(50,75,1001, 'ghz')
f_wr12p2  = Frequency(60,90,1001, 'ghz')
f_wr10  = Frequency(75,110,1001, 'ghz')
f_wr8  = Frequency(90,140,1001, 'ghz')
f_wr6p5  = Frequency(110,170,1001, 'ghz')
f_wr5p1  = Frequency(140,220,1001, 'ghz')
f_wr4p3  = Frequency(170,260,1001, 'ghz')
f_wr3p4  = Frequency(220,330,1001, 'ghz')
f_wr2p8 = Frequency(260,400,1001, 'ghz')
f_wr2p2 = Frequency(330,500,1001, 'ghz')
f_wr1p9 = Frequency(400,600,1001, 'ghz')
f_wr1p5 = Frequency(500,750,1001, 'ghz')
f_wr1p2   = Frequency(600,900,1001, 'ghz')
f_wr1   = Frequency(750,1100,1001, 'ghz')
f_wr0p8   = Frequency(900,1400,1001, 'ghz')
f_wr0p65  = Frequency(1100,1700,1001, 'ghz')
f_wr0p51   = Frequency(1400,2200,1001, 'ghz')

f_wm1295 = f_wr5p1
f_wm1092 = f_wr4p3
f_wm864  = f_wr3p4
f_wm710  = f_wr2p8
f_wm570  = f_wr2p2
f_wm470  = f_wr1p9
f_wm380  = f_wr1p5
f_wm310  = f_wr1p2
f_wm250  = f_wr1
f_wm200  = f_wr0p8
f_wm164  = f_wr0p65
f_wm130  = f_wr0p51


wr51  = RectangularWaveguide(f_wr51.copy(),a=510*mil,b=255*mil,z0=50)
wr42  = RectangularWaveguide(f_wr42.copy(),a=420*mil,b=170*mil,z0=50)
wr34  = RectangularWaveguide(f_wr34.copy(),a=340*mil,b=170*mil,z0=50)
wr28  = RectangularWaveguide(f_wr28.copy(),a=280*mil,b=140*mil,z0=50)
wr22p4  = RectangularWaveguide(f_wr22p4.copy(),a=224*mil,b=112*mil,z0=50)
wr18p8  = RectangularWaveguide(f_wr18p8.copy(),a=188*mil,b=94*mil,z0=50)
wr14p8  = RectangularWaveguide(f_wr14p8.copy(),a=148*mil,b=74*mil,z0=50)
wr12p2  = RectangularWaveguide(f_wr12p2.copy(),a=122*mil,b=61*mil,z0=50)
wr10  = RectangularWaveguide(f_wr10.copy(),a=100*mil,b=50*mil,z0=50)
wr8  = RectangularWaveguide(f_wr8.copy(),a=80*mil,b=40*mil,z0=50)
wr6p5  = RectangularWaveguide(f_wr6p5.copy(),a=65*mil,b=32.5*mil,z0=50)
wr5p1  = RectangularWaveguide(f_wr5p1.copy(),a=51*mil,b=25.5*mil,z0=50)
wr4p3  = RectangularWaveguide(f_wr4p3.copy(),a=43*mil,b=21.5*mil,z0=50)
wr3p4  = RectangularWaveguide(f_wr3p4.copy(),a=34*mil,b=17*mil,z0=50)
wr2p8 = RectangularWaveguide(f_wr2p8.copy(),a=28*mil,b=14*mil,z0=50)
wr2p2 = RectangularWaveguide(f_wr2p2.copy(),a=22*mil,b=11*mil,z0=50)
wr1p9 = RectangularWaveguide(f_wr1p9.copy(),a=19*mil,b=9.5*mil,z0=50)
wr1p5 = RectangularWaveguide(f_wr1p5.copy(),a=15*mil,b=7.5*mil,z0=50)
wr1p2   = RectangularWaveguide(f_wr1p2.copy(),a=12*mil,b=6*mil,z0=50)
wr1   = RectangularWaveguide(f_wr1.copy(),a=10*mil,b=5*mil,z0=50)
wr0p8   = RectangularWaveguide(f_wr0p8.copy(),a=8*mil,b=4*mil,z0=50)
wr0p65  = RectangularWaveguide(f_wr0p65.copy(),a=6.5*mil,b=3.25*mil,z0=50)
wr0p51   = RectangularWaveguide(f_wr0p51.copy(),a=5.1*mil,b=2.55*mil,z0=50)

wm1295 = wr5p1
wm1092 = wr4p3
wm864  = wr3p4
wm710  = wr2p8
wm570  = wr2p2
wm470  = wr1p9
wm380  = wr1p5
wm310  = wr1p2
wm250  = wr1
wm200  = wr0p8
wm164  = wr0p65
wm130  = wr0p51
