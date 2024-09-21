from pathlib import Path

import numpy as np
import pytest

import skrf as rf

test_data = Path(__file__).parent / "ts"


def test_ex_1():
    ts = rf.Network(test_data / "ex_1.ts")


def test_ex_2():
    ts = rf.Network(test_data / "ex_2.ts")
    ref = rf.Network(
        f=np.arange(1,6),
        z=(np.arange(5) + 11) * np.exp(1j*np.arange(10,60,10) * np.pi / 180),
        f_unit="mhz"
                     )
    assert ts == ref

def test_ex_3():
    ts = rf.Network(test_data / "ex_3.ts")
    ref = rf.Network(
        f=[1,2],
        s=[[[111, 112], [121, 122]],
           [[211, 212], [221, 222]]
           ],
        f_unit="ghz"
     )
    assert ts == ref

def test_ex_4():
    ts = rf.Network(test_data / "ex_4.ts")
    ref = rf.Network(
        f=[1],
        s=[[[11, 12, 13, 14],
            [21, 22, 23, 24],
            [31, 32, 33, 34],
            [41, 42, 43, 44]
            ]],
        z0=[50, 75, 0.01, 0.01],
        f_unit="ghz"
     )
    assert ts == ref

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

s = s_mag * np.exp(1j*s_deg * np.pi / 180)
ex_5_6 = rf.Network(s=np.tile(s, [2, 1, 1]), z0=[50, 75, 0.01, 0.01], f=[5e9, 6e9], f_unit="Hz")

@pytest.mark.parametrize("fname",
    [
        test_data / "ex_5.ts",
        test_data / "ex_6.ts",
    ]
)
def test_ts_example_5_6(fname):
    ts = rf.Network(fname)
    assert ex_5_6 == ts

def test_ts_example_7():
    ts = rf.Network(test_data / "ex_7.ts")
    ref = rf.Network(z0=20, f_unit="mhz", f=np.arange(100,600,100),
                     z=[
                         [[74.250*np.exp(1j*( -4) * np.pi / 180)]],
                         [[60.000*np.exp(1j*(-22) * np.pi / 180)]],
                         [[53.025*np.exp(1j*(-45) * np.pi / 180)]],
                         [[30.000*np.exp(1j*(-62) * np.pi / 180)]],
                         [[0.7500*np.exp(1j*(-89) * np.pi / 180)]],
                         ])
    assert ts == ref

def test_example_8():
    ex_7 = rf.Network(test_data / "ex_8.s1p")
    ref = rf.Network(f=[2e6], s=[[[0.894 * np.exp(1j* -12.136 * np.pi / 180)]]])

    assert ex_7 == ref


def test_ts_example_9_10():
    ex_9 = rf.Network(test_data / "ex_9.s1p")
    ex_10 = rf.Network(test_data / "ex_10.ts")

    assert np.allclose(ex_9.z, ex_10.z)

def test_ts_example_11_12():
    ex_11 = rf.Network(test_data / "ex_11.s2p")
    ex_12 = rf.Network(test_data / "ex_12.ts")

    assert ex_11 == ex_12

def test_ts_example_12_12g():
    ex_12 = rf.Network(test_data / "ex_12.ts")
    ex_12_g = rf.Network(test_data / "ex_12_g.ts")

    assert np.allclose(ex_12.s, ex_12_g.s, atol=0.01)

def test_ts_example_13():
    snp = rf.Network(test_data / "ex_13.s2p")
    s = np.array([[[ 3.926e-01-0.1211j, -3.000e-04-0.0021j],
        [-3.000e-04-0.0021j,  3.926e-01-0.1211j]],

       [[ 3.517e-01-0.3054j, -9.600e-03-0.0298j],
        [-9.600e-03-0.0298j,  3.517e-01-0.3054j]],

       [[ 3.419e-01+0.3336j, -1.340e-02+0.0379j],
        [-1.340e-02+0.0379j,  3.419e-01+0.3336j]]])

    ref = rf.Network(s=s, f=[1,2,10], f_unit="ghz")
    assert snp == ref

def test_ts_example_14():
    s_mag = np.array([[[0.6 , 0.4 , 0.42, 0.53],
            [0.4 , 0.6 , 0.53, 0.42],
            [0.42, 0.53, 0.6 , 0.4 ],
            [0.53, 0.42, 0.4 , 0.6 ]],

        [[0.57, 0.4 , 0.41, 0.57],
            [0.4 , 0.57, 0.57, 0.41],
            [0.41, 0.57, 0.57, 0.4 ],
            [0.57, 0.41, 0.4 , 0.57]],

        [[0.5 , 0.45, 0.37, 0.62],
            [0.45, 0.5 , 0.62, 0.37],
            [0.37, 0.62, 0.5 , 0.45],
            [0.62, 0.37, 0.45, 0.5 ]]])

    deg = np.array([[[ 161.24,  -42.2 ,  -66.58,  -79.34],
        [ -42.2 ,  161.2 ,  -79.34,  -66.58],
        [ -66.58,  -79.34,  161.24,  -42.2 ],
        [ -79.34,  -66.58,  -42.2 ,  161.24]],

       [[ 150.37,  -44.34,  -81.24,  -95.77],
        [ -44.34,  150.37,  -95.77,  -81.24],
        [ -81.24,  -95.77,  150.37,  -44.34],
        [ -95.77,  -81.24,  -44.34,  150.37]],

       [[ 136.69,  -46.41,  -99.09, -114.19],
        [ -46.41,  136.69, -114.19,  -99.09],
        [ -99.09, -114.19,  136.69,  -46.41],
        [-114.19,  -99.09,  -46.41,  136.69]]])

    snp = rf.Network(test_data / "ex_14.s4p")
    ref = rf.Network(s=s_mag * np.exp(1j * deg / 180 * np.pi), f=[5,6,7], f_unit="ghz")
    assert snp == ref

def test_ts_example_17():
    s_mag = np.array([
        [[0.95, 0.04],
        [3.57, 0.66]],

       [[0.6 , 0.14],
        [1.3 , 0.56]]])

    s_deg = np.array([
        [[ -26.,   76.],
        [ 157.,  -14.]],

       [[-144.,   40.],
        [  40.,  -85.]]])

    s = s_mag * np.exp(1j*s_deg * np.pi / 180)

    z0 = [[50, 25], [50, 25]]

    ref = rf.Network(f=[2,22], f_unit="GHz", s=s, z0=z0)
    ts = rf.Network(test_data / "ex_17.ts")

    assert ref == ts
    assert ts.noisy

    ts.z0 = 50
    snp = rf.Network(test_data / "ex_18.s2p")
    assert ts == snp
    assert np.allclose(ts.noise, snp.noise)

def test_ts_example_16():
    ts = rf.Network(test_data / "ex_16.ts")
    assert np.all(ts.port_modes == np.array(["S", "D", "C", "S", "D", "C"]))
    assert np.allclose(ts.z0, [50, 150, 37.5, 50, 0.02, 0.005])
