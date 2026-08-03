import io
import zipfile
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


def test_ex_2_version_2_1():
    data = (test_data / "ex_2.ts").read_text().replace("[Version] 2.0", "[Version] 2.1")
    ts = rf.Network.from_string(data)
    ref = rf.Network(
        f=np.arange(1, 6),
        z=(np.arange(5) + 11) * np.exp(1j * np.arange(10, 60, 10) * np.pi / 180),
        f_unit="mhz",
    )

    assert ts == ref


def test_ex_2_write():
    ts = rf.Network(test_data / "ex_2.ts")

    stream = io.StringIO(ts.write_touchstone(return_string=True, form="ma", parameter="Z"))
    stream.name = "ex_2.s1p"
    ts_read_back = rf.Network(stream)

    assert ts == ts_read_back


@pytest.mark.parametrize("version", ["2.0", "2.1"])
def test_ex_2_write_v2(version):
    ts = rf.Network(test_data / "ex_2.ts")

    output = ts.write_touchstone(return_string=True, form="ma", parameter="Z", version=version)
    ts_read_back = rf.Network.from_string(output)

    assert f"[Version] {version}" in output
    assert "[Number of Ports] 1" in output
    assert "[Number of Frequencies] 5" in output
    assert "[Network Data]" in output
    assert output.rstrip().endswith("[End]")
    assert ts == ts_read_back


@pytest.mark.parametrize("parameter", ["Y", "Z", "G", "H"])
def test_write_v2_non_s_parameters_are_unnormalized(parameter):
    ts = rf.Network(test_data / "ex_12.ts")
    ts.renormalize([45 + 2j, 60 + 3j])

    output = ts.write_touchstone(return_string=True, form="ri", parameter=parameter, version="2.1")
    ts_read_back = rf.Network.from_string(output)

    assert "[Reference]" not in output
    np.testing.assert_allclose(getattr(ts, parameter.lower()), getattr(ts_read_back, parameter.lower()))


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


@pytest.mark.parametrize("version", ["2.0", "2.1"])
def test_ex_4_write_v2(version):
    ts = rf.Network(test_data / "ex_4.ts")

    output = ts.write_touchstone(return_string=True, form="ma", version=version)
    ts_read_back = rf.Network.from_string(output)

    assert f"[Version] {version}" in output
    assert "[Reference] 50.0 75.0 0.01 0.01" in output
    assert ts == ts_read_back


def test_write_v2_renormalizes_frequency_dependent_z0():
    ts = rf.Network(
        f=[1, 2],
        f_unit="ghz",
        s=np.zeros((2, 2, 2), dtype=complex),
        z0=[[50, 75], [55, 80]],
        name="frequency_dependent_z0",
    )

    with pytest.raises(ValueError, match="vary with frequency"):
        ts.write_touchstone(return_string=True, version="2.1")

    output = ts.write_touchstone(return_string=True, version="2.1", r_ref=50)
    ts_read_back = rf.Network.from_string(output)
    expected = ts.copy()
    expected.renormalize(50)

    np.testing.assert_allclose(expected.s, ts_read_back.s)
    np.testing.assert_allclose(expected.z0, ts_read_back.z0)


def test_write_v2_with_hfss_port_impedance_comments():
    ts = rf.Network(
        f=[1, 2],
        f_unit="ghz",
        s=np.zeros((2, 2, 2), dtype=complex),
        z0=[[50 + 1j, 75 + 2j], [55 + 3j, 80 + 4j]],
        name="complex_z0",
        s_def="traveling",
    )

    output = ts.write_touchstone(return_string=True, version="2.1", write_z0=True)
    ts_read_back = rf.Network.from_string(output)

    assert "! Port Impedance" in output
    np.testing.assert_allclose(ts.s, ts_read_back.s)
    np.testing.assert_allclose(ts.z0, ts_read_back.z0)


def test_write_v2_rejects_invalid_inputs():
    ts = rf.Network(f=[1], s=[[[0]]], name="invalid")

    with pytest.raises(ValueError, match="version"):
        ts.write_touchstone(return_string=True, version="3.0")
    with pytest.raises(ValueError, match="positive"):
        ts.copy().write_touchstone(return_string=True, version="2.1", r_ref=-50)
    with pytest.raises(ValueError, match="at least one frequency"):
        rf.Network(name="empty").write_touchstone(return_string=True, version="2.1")

    mixed_mode = ts.copy()
    mixed_mode.port_modes = np.array(["D"])
    with pytest.raises(NotImplementedError, match="mixed-mode"):
        mixed_mode.write_touchstone(return_string=True, version="2.1")


@pytest.mark.parametrize(
    ("encoding", "encoded_comment"),
    [
        ("ISO-8859-1", b"\xb5"),
        ("UTF-8", b"\xc2\xb5"),
    ],
)
def test_write_v2_file_encoding(tmp_path, encoding, encoded_comment):
    ts = rf.Network(test_data / "ex_4.ts")
    ts.name = "encoded"
    ts.comments = "micro sign: µ"
    filename = tmp_path / ts.name

    ts.write_touchstone(filename, version="2.1", encoding=encoding)

    output_file = filename.with_suffix(".ts")
    output_bytes = output_file.read_bytes()
    explicit_encoding = rf.Network(output_file, encoding=encoding)
    detected_encoding = rf.Network(output_file)

    assert encoded_comment in output_bytes
    assert "micro sign: µ" in explicit_encoding.comments
    assert "micro sign: µ" in detected_encoding.comments
    assert ts == explicit_encoding
    assert ts == detected_encoding


def test_write_touchstone_defaults_to_iso_8859_1(tmp_path):
    ts = rf.Network(test_data / "ex_4.ts")
    ts.comments = "micro sign: µ"
    filename = tmp_path / "default_encoding"

    ts.write_touchstone(filename, version="2.1")

    output_bytes = filename.with_suffix(".ts").read_bytes()
    assert b"\xb5" in output_bytes
    assert b"\xc2\xb5" not in output_bytes


def test_read_touchstone_honors_explicit_encoding(tmp_path):
    ts = rf.Network(test_data / "ex_4.ts")
    ts.comments = "micro sign: µ"
    filename = tmp_path / "latin1"
    ts.write_touchstone(filename, version="2.1")

    with pytest.raises(UnicodeDecodeError):
        rf.Network(filename.with_suffix(".ts"), encoding="UTF-8")


@pytest.mark.parametrize(
    ("encoding", "encoded_comment"),
    [
        ("ISO-8859-1", b"\xb5"),
        ("UTF-8", b"\xc2\xb5"),
    ],
)
def test_write_v2_zip_encoding(tmp_path, encoding, encoded_comment):
    ts = rf.Network(test_data / "ex_4.ts")
    ts.name = "encoded"
    ts.comments = "micro sign: µ"
    archive_path = tmp_path / "touchstones.zip"

    with zipfile.ZipFile(archive_path, "w") as archive:
        ts.write_touchstone("encoded.ts", to_archive=archive, version="2.1", encoding=encoding)
        archive.writestr("reports", "not a Touchstone file")

    with zipfile.ZipFile(archive_path) as archive:
        output_bytes = archive.read("encoded.ts")
        explicit_encoding = rf.Network.zipped_touchstone("encoded.ts", archive, encoding=encoding)
        detected_encoding = rf.Network.zipped_touchstone("encoded.ts", archive)
        discovered_networks = rf.io.read_zipped_touchstones(archive, encoding=encoding)
        auto_discovered_networks = rf.io.read_zipped_touchstones(archive)

    assert encoded_comment in output_bytes
    assert "micro sign: µ" in explicit_encoding.comments
    assert "micro sign: µ" in detected_encoding.comments
    assert ts == explicit_encoding
    assert ts == detected_encoding
    assert discovered_networks == {"encoded": detected_encoding}
    assert auto_discovered_networks == {"encoded": detected_encoding}


@pytest.mark.parametrize("encoding", ["UTF-16", "ASCII", "not-a-codec"])
def test_write_touchstone_rejects_unsupported_encoding(encoding):
    ts = rf.Network(test_data / "ex_4.ts")

    with pytest.raises(ValueError, match="encoding"):
        ts.write_touchstone(return_string=True, version="2.1", encoding=encoding)


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

def test_ex_9_write():
    ts = rf.Network(test_data / "ex_9.s1p")

    stream = io.StringIO(ts.write_touchstone(return_string=True, form="ma", parameter="Z"))
    stream.name = "ex_9.s1p"
    ts_read_back = rf.Network(stream)

    assert ts == ts_read_back

def test_ex_10_write():
    ts = rf.Network(test_data / "ex_10.ts")

    stream = io.StringIO(ts.write_touchstone(return_string=True, form="ma", parameter="Z"))
    stream.name = "ex_10.s1p"
    ts_read_back = rf.Network(stream)

    assert ts == ts_read_back

def test_ts_example_11_12():
    ex_11 = rf.Network(test_data / "ex_11.s2p")
    ex_12 = rf.Network(test_data / "ex_12.ts")

    assert ex_11 == ex_12

def test_ts_example_12_12g():
    ex_12 = rf.Network(test_data / "ex_12.ts")
    ex_12_g = rf.Network(test_data / "ex_12_g.ts")

    assert np.allclose(ex_12.s, ex_12_g.s, atol=0.01)

def test_ex_12_g_write():
    ts = rf.Network(test_data / "ex_12_g.ts")

    stream = io.StringIO(ts.write_touchstone(return_string=True, form="ma", parameter="G"))
    stream.name = "ex_12_g.s2p"
    ts_read_back = rf.Network(stream)

    assert ts == ts_read_back

def test_ex_12_h_write():
    ts = rf.Network(test_data / "ex_12.ts")

    stream = io.StringIO(ts.write_touchstone(return_string=True, form="ma", parameter="H"))
    stream.name = "ex_12.s2p"
    ts_read_back = rf.Network(stream)

    assert ts == ts_read_back

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


@pytest.mark.parametrize("version", ["2.0", "2.1"])
def test_ts_example_17_write_v2(version):
    ts = rf.Network(test_data / "ex_17.ts")

    output = ts.write_touchstone(return_string=True, form="ma", version=version)
    ts_read_back = rf.Network.from_string(output)

    assert "[Two-Port Data Order] 21_12" in output
    assert "[Number of Noise Frequencies] 2" in output
    assert "[Reference] 50.0 25.0" in output
    assert "[Noise Data]" in output
    noise_block = output.partition("[Noise Data]\n")[2].partition("[End]")[0]
    noise_resistances = [
        float(line.split()[-1])
        for line in noise_block.splitlines()
        if line and not line.startswith("!")
    ]
    np.testing.assert_allclose(noise_resistances, [19, 20])
    np.testing.assert_allclose(ts.s, ts_read_back.s)
    np.testing.assert_allclose(ts.z0, ts_read_back.z0)
    expected_noise = ts.copy()
    expected_noise.resample(ts.f_noise)
    ts_read_back.resample(ts_read_back.f_noise)
    np.testing.assert_allclose(expected_noise.nfmin_db, ts_read_back.nfmin_db)
    np.testing.assert_allclose(expected_noise.g_opt, ts_read_back.g_opt)
    np.testing.assert_allclose(expected_noise.rn, ts_read_back.rn)


def test_ts_example_16():
    ts = rf.Network(test_data / "ex_16.ts")
    assert np.all(ts.port_modes == np.array(["S", "D", "C", "S", "D", "C"]))
    assert np.allclose(ts.z0, [50, 150, 37.5, 50, 0.02, 0.005])
