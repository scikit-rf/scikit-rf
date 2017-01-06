from collections import OrderedDict
import importlib
import glob
import os.path
import traceback
import sys

from . import base_analyzer

this_path = os.path.normpath(os.path.dirname(__file__))
analyzer_modules = glob.glob(this_path + "/analyzer_*.py")
analyzers = OrderedDict()

sys.path.insert(0, this_path)
for analyzer in analyzer_modules:
    module_name = os.path.basename(analyzer)[:-3]

    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        etype, value, tb = sys.exc_info()
        err_msg = "\n".join(traceback.format_exception(etype, value, tb))
        print("did not import {:s}\n\n{:s}".format(module_name, err_msg))
        continue

    if module.Analyzer.NAME in analyzers.keys():
        print("overwriting Analyzer {:s} in selection".format(module.Analyzer.NAME))

    analyzers[module.Analyzer.NAME] = module.Analyzer
sys.path.pop(0)
