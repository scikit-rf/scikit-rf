'''
this is just a hack to allow ivi Drivers to be back-ward compatable 
with the pyvisa GpibInstrument class. 

Efforts are underway for a unification of visa/vxi and things by the 
group https://github.com/LabPy, so this hack will go way and the 
instrument backends will be cleaned up soon. 


'''
from ivi import Driver

try:
    # rename the ivi method so our legacy VI's still work
    Driver.ask_for_values = Driver._ask_for_values
except:
    # if they dont have git version of python-ivi 
    def ask_for_values(self,msg, delim=',', converter=float, array = True):
        s = self._ask(msg)
        s_split = s.split(delim)
        out = map(converter, s_split)
        if array:
            out = npy.array(out)
        return out
    Driver.ask_for_values = ask_for_values
    
Driver.ask = Driver._ask
Driver.write = Driver._write
