import numpy as np
from scipy.optimize.linesearch import scalar_search_wolfe2


class LineSearchTool(object):

    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        elif self._method == 'Best':
            pass
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        phi = lambda a: oracle.func_directional(x_k, d_k, a)
        derphi = lambda a: oracle.grad_directional(x_k, d_k, a)
        phi0, derphi0 = phi(0), derphi(0)

        if self._method == 'Constant':
            return self.c

        if self._method == 'Wolfe':
            alpha, _, _, _ = scalar_search_wolfe2(phi=phi, derphi=derphi,
                                                  phi0=phi0, derphi0=derphi0,
                                                  c1=self.c1, c2=self.c2)
            if alpha:
                return alpha
            else:
                return LineSearchTool(method='Armijo', c1=self.c1, alpha=self.alpha_0).line_search(oracle, x_k, d_k,
                                                                                                   previous_alpha)

        if self._method == 'Armijo':
            alpha = previous_alpha if previous_alpha else self.alpha_0
            while phi(alpha) > phi0 + self.c1 * alpha * derphi0:
                alpha /= 2
            return alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()
