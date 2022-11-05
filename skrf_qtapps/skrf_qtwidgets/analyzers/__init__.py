from collections import OrderedDict
import importlib
import glob
import os.path
import traceback
import sys

analyzers = OrderedDict()

try:
    from skrf.vi.vna import keysight_pna

    this_path = os.path.normpath(os.path.dirname(__file__))
    analyzer_modules = glob.glob(this_path + "/analyzer_*.py")
    analyzers[keysight_pna.PNA.NAME] = keysight_pna.PNA

    sys.path.insert(0, this_path)
    for analyzer in analyzer_modules:
        module_name = os.path.basename(analyzer)[:-3]

        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            etype, value, tb = sys.exc_info()
            err_msg = "\n".join(traceback.format_exception(etype, value, tb))
            print(f"did not import {module_name:s}\n\n{err_msg:s}")
            continue

        if module.Analyzer.NAME in analyzers.keys():
            print(f"overwriting Analyzer {module.Analyzer.NAME:s} in selection")

        analyzers[module.Analyzer.NAME] = module.Analyzer
    sys.path.pop(0)
except ImportError:
    pass


