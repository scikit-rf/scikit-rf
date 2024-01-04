import skrf as rf
import numpy as np
from pathlib import Path

test_data = Path(__file__).parent / "ts" 

def test_ts_example_5():
    ts = rf.Network(test_data / "ex_5.ts")

    s_mag = np.array(
        [[[0.6 , 0.4 , 0.42, 0.53],
        [0.4 , 0.6 , 0.53, 0.42],
        [0.42, 0.53, 0.6 , 0.4 ],
        [0.53, 0.42, 0.4 , 0.6 ]]])
    
    s_deg = np.array(
        [[[161.24, -42.2 , -66.58, -79.34],
        [-42.2 , 161.2 , -79.34, -66.58],
        [-66.58, -79.34, 161.24, -42.2 ],
        [-79.34, -66.58, -42.2 , 161.24]]])
    
    s = s_mag * np.exp(1j*s_deg*np.pi/180)

    ref = rf.Network(name="ex_5", s=s, z0=[50, 75, 0.01, 0.01], f=5e9, f_unit="Hz")
    assert ref == ts

def test_ts_example_6():
    ts = rf.Network(test_data / "ex_6.ts")

    s_mag = np.array(
        [[[0.6 , 0.4 , 0.42, 0.53],
        [0.4 , 0.6 , 0.53, 0.42],
        [0.42, 0.53, 0.6 , 0.4 ],
        [0.53, 0.42, 0.4 , 0.6 ]]])
    
    s_deg = np.array(
        [[[161.24, -42.2 , -66.58, -79.34],
        [-42.2 , 161.2 , -79.34, -66.58],
        [-66.58, -79.34, 161.24, -42.2 ],
        [-79.34, -66.58, -42.2 , 161.24]]])
    
    s = s_mag * np.exp(1j*s_deg*np.pi/180)

    ref = rf.Network(name="ex_6", s=s, z0=[50, 75, 0.01, 0.01], f=5e9, f_unit="Hz")

    assert ref == ts