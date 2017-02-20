from skrf.vi.scpi.parser import parse_yaml_file

yaml_files = [
    'keysight_pna_scpi.yaml'
]

for fname in yaml_files:
    parse_yaml_file(fname)
