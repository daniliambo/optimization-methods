import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from scipy.optimize.linesearch import scalar_search_wolfe2
from scipy.linalg import cholesky, cho_solve
from collections import defaultdict
import time


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """

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
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == 'Constant':
            return self.c

        def f(alp):
            return oracle.func_directional(x_k, d_k, alp)

        def df(alp):
            return oracle.grad_directional(x_k, d_k, alp)

        alpha = previous_alpha if previous_alpha is not None else self.alpha_0

        if self._method == 'Wolfe':
            ret_wolf = scalar_search_wolfe2(f, derphi=df, c1=self.c1, c2=self.c2)
            a = ret_wolf[0]
            if a is None:
                a = self._backtracking(f, df, alpha)
        elif self._method == 'Armijo':
            a = self._backtracking(f, df, alpha)

        return a

    def _backtracking(self, f, df, alp):
        f_0 = f(0)
        df_0 = df(0)
        while f(alp) > f_0 + self.c1 * alp * df_0:
            alp = alp / 2

        return alp


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    ts = time.time()
    x_k = np.copy(x_0)
    df_0 = oracle.grad(x_k)
    df_k = np.copy(df_0)
    norm_0 = np.dot(df_0, df_0)

    if trace:
        time_diff = time.time() - ts
        add_to_hist(history,
                    time=time_diff,
                    func=oracle.func(x_k),
                    norm=norm_0,
                    x_k=x_k,
                    norm_to_start=norm_0 / norm_0
                    )

    for i in range(max_iter + 1):
        alpha = line_search_tool.line_search(oracle, x_k, -df_k)

        x_k = x_k - alpha * df_k
        df_k = oracle.grad(x_k)
        norm_k = np.dot(df_k, df_k)

        if trace:
            time_diff = time.time() - ts
            add_to_hist(history,
                        time=time_diff,
                        func=oracle.func(x_k),
                        norm=norm_k,
                        x_k=x_k,
                        norm_to_start=norm_k/norm_0
                        )

        if not np.all(np.isfinite(df_k)) or not np.all(np.isfinite(norm_k)) or not np.all(np.isfinite(x_k)):
            return x_k, 'computational_error', history

        if norm_k <= tolerance * norm_0:
            return x_k, 'success', history

    return x_k, 'iterations_exceeded', history


def add_to_hist(hist, time, func, norm, x_k, norm_to_start):
    hist['time'].append(time)
    hist['func'].append(func)
    hist['grad_norm'].append(norm)
    hist['x'].append(x_k)
    hist['norm_to_start'].append(norm_to_start)

    return hist


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    ts = time.time()
    x_k = np.copy(x_0)
    df_0 = oracle.grad(x_k)
    grad_norm_squared_0 = np.dot(df_0, df_0)

    hess_k = oracle.hess(x_k)
    dk = _find_direction_newton(hess_k, df_0)

    if trace:
        time_diff = time.time() - ts
        add_to_hist(history,
                    time=time_diff,
                    func=oracle.func(x_k),
                    norm=grad_norm_squared_0,
                    x_k=x_k,
                    norm_to_start=grad_norm_squared_0 / grad_norm_squared_0
                    )

    for i in range(max_iter + 1):

        alpha = line_search_tool.line_search(oracle, x_0, dk)
        x_k = x_k + alpha * dk

        df_k = oracle.grad(x_k)
        grad_norm_squared_k = np.dot(df_k, df_k)
        hess_k = oracle.hess(x_k)
        dk = _find_direction_newton(hess_k, df_k)

        if trace:
            time_diff = time.time() - ts
            add_to_hist(history,
                        time=time_diff,
                        func=oracle.func(x_k),
                        norm=grad_norm_squared_k,
                        x_k=x_k,
                        norm_to_start=grad_norm_squared_k / grad_norm_squared_0
                        )

        if not np.all(np.isfinite(df_k)) \
                or not np.all(np.isfinite(grad_norm_squared_k)) \
                or not np.all(np.isfinite(x_k)):
            return x_k, 'computational_error', history

        if grad_norm_squared_k <= tolerance * grad_norm_squared_0:
            return x_k, 'success', history


    return x_k, 'iterations_exceeded', history


def _find_direction_newton(A, grad):
    U = cholesky(A)
    dk = cho_solve((U, False), -grad)

    return dk


if __name__ == '__main__':
    from oracles import QuadraticOracle

    A = np.eye(2) * 2
    b = np.array([2., 2.])
    oracle = QuadraticOracle(A, b)
    print(oracle.func(np.array([1, 1])))
    # print(oracle.grad([0.5, 0.5]))
    # print(oracle.hess([0.5, 0.5]))
    ls = {'method': 'Armijo', 'c1': 1e-4}
    # print(gradient_descent(oracle, np.array([5, 5]), line_search_options=ls))

    oracle = QuadraticOracle(np.eye(5), np.arange(5))
    print(oracle.func(np.array([0, 1, 2, 3, 4])))
    x_opt, message, history = gradient_descent(oracle, np.zeros(5),
                                                  line_search_options={'method': 'Wolfe', 'c1': 1e-4})

    print('Found optimal point: {}, {}'.format(x_opt, message))
