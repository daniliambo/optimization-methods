import numpy as np
from numpy.linalg import LinAlgError
import scipy
import time
from datetime import datetime
from collections import defaultdict


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

        # TODO: Implement line search procedures for Armijo, Wolfe and Constant steps.
        def armijo():
            a = previous_alpha or self.alpha_0
            while oracle.func_directional(x_k, d_k, a) \
                    > oracle.func_directional(x_k, d_k, 0) \
                    + self.c1 * a * oracle.grad_directional(x_k, d_k, 0):
                a /= 2
            return a

        if self._method == 'Wolfe':
            a = previous_alpha or self.alpha_0
            from scipy.optimize.linesearch import scalar_search_wolfe2

            a, *_ = scalar_search_wolfe2(lambda x: oracle.func_directional(x, d_k, a),
                                         lambda x: oracle.grad_directional(x, d_k, a),
                                         oracle.func_directional(0, d_k, a), None,
                                         oracle.grad_directional(0, d_k, a),
                                         self.c1,
                                         self.c2)
            if a:
                return a
            return LineSearchTool(method='Armijo', c1=self.c1, alpha_0=self.alpha_0).line_search(oracle, x_k, d_k,
                                                                                                 previous_alpha)

        elif self._method == 'Armijo':
            return armijo()

        elif self._method == 'Constant':
            return self.c


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def checkpoint(x, d, oracle, history, start_time):
    history['time'].append(time.time() - start_time)
    history['func'].append(oracle.func(x))
    history['grad_norm'].append(np.linalg.norm(d))
    if x.size <= 2:
        history['x'].append(x)


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
    x_k, a_k = np.copy(x_0), None

    # TODO: Implement gradient descent
    # Use line_search_tool.line_search() for adaptive step size.

    grad_0_norm = np.linalg.norm(oracle.grad(x_0)) ** 2
    stop = tolerance * grad_0_norm
    if trace:
        history['func'].append(oracle.func(x_k))
        history['time'].append(0.0)
        history['grad_norm'].append(grad_0_norm)

    try:
        start_time = time.time()
        for it in range(max_iter):

            # performance
            d_k = oracle.grad(x_k)
            if np.linalg.norm(d_k) ** 2 <= stop:
                if display:
                    print("iteration: {}\ntime: {}\ngrad_norm: {}\n\n".format(it, time.time() - start_time,
                                                                              np.linalg.norm(d_k) ** 2))
                if trace:
                    checkpoint(x_k, d_k, oracle, history, start_time)
                return x_k, 'success', history
            else:
                a_k = line_search_tool.line_search(oracle, x_k, -d_k, 2 * a_k if a_k else None)
                x_k = x_k - a_k * d_k

            if display:
                print("iteration: {}\ntime: {}\ngrad_norm: {}\n\n".format(it, time.time() - start_time,
                                                                          np.linalg.norm(d_k)))
            if trace:
                checkpoint(x_k, d_k, oracle, history, start_time)

        else:
            return x_k, 'iterations_exceeded', history
    except BaseException as e:
        print(e)
        return x_k, 'computational_error', history


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
    x_k = np.copy(x_0)

    # TODO: Implement Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    a_k = None

    # TODO: Implement Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.

    grad_0_norm = np.linalg.norm(oracle.grad(x_0)) ** 2
    stop = tolerance * grad_0_norm

    if trace:
        history['func'].append(oracle.func(x_k))
        history['time'].append(0.0)
        history['grad_norm'].append(grad_0_norm)

    try:
        start_time = time.time()
        for it in range(max_iter):

            # debug
            if display:
                print(it)

            # performance
            g_k = oracle.grad(x_k)
            try:
                h_k = oracle.hess(x_k)
                d_k = scipy.linalg.cho_solve(scipy.linalg.cho_factor(h_k), g_k)
            except LinAlgError:
                return x_k, 'newton_direction_error', history

            if np.linalg.norm(d_k) ** 2 <= stop:
                if trace:
                    checkpoint(x_k, d_k, oracle, history, start_time)
                return x_k, 'success', history
            else:
                a_k = line_search_tool.line_search(oracle, x_k, -d_k, a_k if a_k else None)
                x_k = x_k - a_k * d_k

            # history
            if trace:
                checkpoint(x_k, d_k, oracle, history, start_time)
        else:
            return x_k, 'iterations_exceeded', history
    except BaseException as e:
        print(e)
        return x_k, 'computational_error', history


def gradient_descent_(oracle, x_0, tolerance=1e-5, max_iter=10000,
                      line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k, a_k = np.copy(x_0), None

    grad_0_norm = np.linalg.norm(oracle.grad(x_0)) ** 2
    stop = tolerance * grad_0_norm
    if trace:
        history['func'].append(oracle.func(x_k))
        history['time'].append(0.0)
        history['grad_norm'].append(grad_0_norm)

    try:
        start_time = time.time()
        for it in range(max_iter):

            # performance
            d_k = oracle.grad(x_k)
            if np.linalg.norm(d_k) ** 2 <= stop:
                if display:
                    print("iteration: {}\ntime: {}\ngrad_norm: {}\n\n".format(it, time.time() - start_time,
                                                                              np.linalg.norm(d_k) ** 2))
                if trace:
                    checkpoint(x_k, d_k, oracle, history, start_time)
                return x_k, 'success', history
            else:
                a_k = line_search_tool.line_search(oracle, x_k, -d_k, 2 * a_k if a_k else None)
                x_k = x_k - a_k * d_k

            if display:
                print("iteration: {}\ntime: {}\ngrad_norm: {}\n\n".format(it, time.time() - start_time,
                                                                          np.linalg.norm(d_k)))
            if trace:
                checkpoint(x_k, d_k, oracle, history, start_time)

        else:
            return x_k, 'iterations_exceeded', history
    except BaseException as e:
        print(e)
        return x_k, 'computational_error', history


def conjugate_gradients_dev(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax = b using the Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. If max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    # gradient decent ---------------------------------
    history = defaultdict(list) if trace else None
    x_k, a_k = np.copy(x_0), None

    grad_0_norm = np.linalg.norm(matvec(x_0)) ** 2
    stop = tolerance * grad_0_norm

    if trace:
        history['func'].append(matvec(x_k))
        history['time'].append(0.0)
        history['grad_norm'].append(grad_0_norm)
    try:
        start_time = time.time()
        for it in range(max_iter):

            # performance
            d_k = oracle.grad(x_k)
            if np.linalg.norm(d_k) ** 2 <= stop:
                if display:
                    print("iteration: {}\ntime: {}\ngrad_norm: {}\n\n".format(it, time.time() - start_time,
                                                                              np.linalg.norm(d_k) ** 2))
                if trace:
                    checkpoint(x_k, d_k, oracle, history, start_time)
                return x_k, 'success', history
            else:
                a_k = line_search_tool.line_search(oracle, x_k, -d_k, 2 * a_k if a_k else None)
                x_k = x_k - a_k * d_k

            if display:
                print("iteration: {}\ntime: {}\ngrad_norm: {}\n\n".format(it, time.time() - start_time,
                                                                          np.linalg.norm(d_k)))
            if trace:
                checkpoint(x_k, d_k, oracle, history, start_time)

        else:
            return x_k, 'iterations_exceeded', history
    except BaseException as e:
        print(e)
        return x_k, 'computational_error', history

    # gradient descent --------------------------------------------

    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    r_k = b - matvec(x_k)  # Calculate initial residual vector
    d_k = np.copy(r_k)  # Initialize descent direction
    k = 0

    # Initialize helpful variables
    alpha = 0
    beta = 0

    if max_iter is None or max_iter > len(x_0):
        max_iter = len(x_0)  # Sets maximum number of iterations to the dimension of the vector x

    # Define start_time variable
    start_time = time.time()

    # Start of the optimization loop
    while k < max_iter and np.linalg.norm(r_k) >= tolerance * np.linalg.norm(b):
        # Calculate matrix vector product
        q_k = matvec(d_k)
        # Calculate step size for given iteration
        alpha = r_k.dot(r_k) / d_k.dot(q_k)
        # Use the step size to calculate the new approximation
        x_k = x_k + alpha * d_k
        # Calculate the new residual vector
        r_k_1 = r_k - alpha * q_k
        # Calculate the new beta parameter
        beta = r_k_1.dot(r_k_1) / r_k.dot(r_k)
        # Calculate the new descent direction using the new beta parameter
        d_k = r_k_1 + beta * d_k

        # Store history entry
        if trace:
            history['time'].append(time.time() - start_time)
            history['residual_norm'].append(np.linalg.norm(r_k))
            if x_0.size <= 2:
                history['x'].append(np.copy(x_k))

        # Set new values
        r_k = np.copy(r_k_1)
        k += 1

    # End of optimization loop
    if np.linalg.norm(r_k) <= tolerance * np.linalg.norm(b):
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history
