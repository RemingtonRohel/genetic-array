import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Global values that will be used by other scripts
matplotlib.rcParams['backend'] = 'Agg'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'cm'

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

param_dict = {'font.size': SMALL_SIZE,          # controls default text sizes
              'axes.titlesize': MEDIUM_SIZE,    # fontsize of the axes title
              'axes.labelsize': MEDIUM_SIZE,    # fontsize of the x and y labels
              'xtick.labelsize': SMALL_SIZE,    # fontsize of the tick labels
              'ytick.labelsize': SMALL_SIZE,    # fontsize of the tick labels
              'legend.fontsize': SMALL_SIZE,    # legend fontsize
              'figure.titlesize': BIGGER_SIZE,  # fontsize of the figure title
              # 'figure.dpi': 300.0
              }
matplotlib.rcParams.update(param_dict)


def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    """Copied from
    https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib"""
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r'$0$'
            if num == 1:
                return r'$%s$' % latex
            elif num == -1:
                return r'$-%s$' % latex
            else:
                return r'$%s%s$' % (num, latex)
        else:
            if num == 1:
                return r'$\frac{%s}{%s}$' % (latex, den)
            elif num == -1:
                return r'$\frac{-%s}{%s}$' % (latex, den)
            else:
                return r'$\frac{%s%s}{%s}$' % (num, latex, den)

    return _multiple_formatter


class Multiple:
    """Copied from
        https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib"""
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))

