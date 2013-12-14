

'''

.. currentmodule:: skrf.constants
========================================
constants (:mod:`skrf.constants`)
========================================

This module contains pre-initialized objects's. 



Standard Waveguide Bands
----------------------------

:class:`~skrf.frequency.Frequency` Objects
++++++++++++++++++++++++++++++++++++++++++++
These are predefined :class:`~skrf.frequency.Frequency` objects
that correspond to standard waveguide bands. This information is taken
from the VDI Application Note 1002 [#]_ . 


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

Shorthand Names 
----------------

Below is a list of shorthand object names which can be use to save some 
typing. These names are defined in the main `__init__` module.


============ ================
Shorthand    Full Object Name   
============ ================
F            :class:`~skrf.frequency.Frequency`
N            :class:`~skrf.network.Network`
NS           :class:`~skrf.networkSet.NetworkSet`
M            :class:`~skrf.media.media.Media`
C            :class:`~skrf.calibration.calibration.Calibration`
============ ================

The following are shorthand names for commonly used, but unfortunately
longwinded functions.

============ ================
Shorthand    Full Object Name   
============ ================
saf          :func:`~skrf.util.save_all_figs`
============ ================
 



References
-------------
.. [#] VDI Application Note:  VDI Waveguide Band Designations (VDI-1002) http://vadiodes.com/VDI/pdf/waveguidechart200908.pdf
'''



from frequency import Frequency
from media import RectangularWaveguide, Media

from scipy.constants import c, micron, mil, inch, centi, milli, nano, micro,pi



# globals 


# pre-initialized classes       
        
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


