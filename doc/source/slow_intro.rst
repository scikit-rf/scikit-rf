.. _slow-intro:

Slow Introduction
**********************


This is a slow  introduction to **skrf** for readers who arent especially familiar with python. If you are familiar with python, or are impatient see the :doc:`introduction`.

**skrf**, like all of python, can be used in scripts or through the python interpreter. If you are new to python and don't understand anything on this page, please see the Install page first.
From a python shell or similar (ie IPython),  the **skrf** module can be imported like so::

	import skrf as rf


From here all **skrf**'s functions can be accessed through the variable 'mv'. Help can be accessed through pythons help command. For example, to get help with the Network class ::
	
	help(rf.Network) 

The Network class is a representation of a n-port network. The most common way to initialize a Network is by loading data saved in a touchstone file. Touchstone files have the extension '.sNp', where N is the number of ports of the network. 
To create a Network from the touchstone file 'horn.s1p'::
	
	horn = rf.Network('horn.s1p')

	

From here you can tab out the contents of the newly created Network by typing ``horn.[hit tab]``. You can get help on the various functions as described above.  The base storage format for a Network's data is in scattering parameters, these can be accessed by the property, 's'. Basic element-wise arithmetic can also be done on the scattering parameters, through operations on the Networks themselves. For instance if you want to form the complex division of two Networks scatering matrices, 


This can also be used to implement averaging


Other non-elementwise operations are also available, such as cascading and de-embeding two-port networks. For instance the composit network of two, two-port networks is formed using the power operator (``**``), 


De-embeding can be accomplished by using the floor division (``//``) operator 
