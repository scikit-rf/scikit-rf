## CSS settings. 
# style sheet was taken from 
# https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers

import json
s = json.load( open("./styles/matplotlibrc.json") )
matplotlib.rcParams.update(s)

from IPython.core.display import HTML


def css_styling():
    styles = open("./styles/custom.css", "r").read()
    return HTML(styles)
css_styling()

