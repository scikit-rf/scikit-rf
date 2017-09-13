from skrf.vi.scpi.parser import parse_yaml_file

yaml_files = [
    'keysight_pna_scpi.yaml',
    'keysight_fieldfox_scpi.yaml',
    'rs_zva_scpi.yaml'
]

for fname in yaml_files:
    print("parsing: " + fname)
    parse_yaml_file(fname)
