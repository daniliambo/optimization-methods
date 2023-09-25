from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve
import scipy.linalg as sli
import scipy.optimize.linesearch as ls
import time
from datetime import datetime


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

        phi = lambda a: oracle.func_directional(x_k, d_k, a)
        dphi = lambda a: oracle.grad_directional(x_k, d_k, a)
        if self._method != 'Constant':
            alpha = self.alpha_0 if previous_alpha is None else previous_alpha

        def backtracking(a_0):
            phi_0 = phi(0)
            dphi_0 = dphi(0)
            while phi(a_0) > phi_0 + self.c1 * a_0 * dphi_0:
                a_0 /= 2
            return a_0

        sp = lambda x: np.array_split(x, 2)  # x -> x[:half], x[half:]

        y_up, y_down = sp(x_k)
        d_up, d_down = sp(d_k)

        try:
            alpha_max_1 = np.min(((y_up - y_down) / (d_down - d_up))[d_down - d_up < 0])
        except:
            alpha_max_1 = self.alpha_0

        try:
            alpha_max_2 = np.min((-(y_up + y_down) / (d_up + d_down))[d_down + d_up < 0])
        except:
            alpha_max_2 = self.alpha_0

        alpha = min(alpha, alpha_max_1 * 0.99, alpha_max_2 * 0.99)

        if self._method == 'Armijo':
            return backtracking(alpha)
        elif self._method == 'Wolfe':
            a_wolf, b, bb, bbb = ls.scalar_search_wolfe2(phi, derphi=dphi, c1=self.c1, c2=self.c2)
            if a_wolf is None:
                return backtracking(alpha)
            else:
                return a_wolf
        elif self._method == 'Constant':
            return self.c
        return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    start = time.time()

    def pushHistory(x_k, oracle, df_k_n):
        if trace:
            history['time'].append(time.time() - start)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(df_k_n ** (1 / 2))
            if np.alen(x_k) <= 2:
                history['x'].append(np.copy(x_k))
        if display:
            print(x_k)

    df0 = oracle.grad(x_0)
    df0_norm = df0.dot(df0)
    for k in range(max_iter):
        df_k = oracle.grad(x_k)
        hf_k = oracle.hess(x_k)
        df_k_norm = df_k.dot(df_k)
        pushHistory(x_k, oracle, df_k_norm)
        if df_k_norm <= df0_norm * tolerance:
            return x_k, 'success', history
        try:
            d_k = sli.cho_solve(sli.cho_factor(hf_k), df_k * (-1))
        except sli.LinAlgError as e:
            return x_k, 'newton_direction_error', history
        a_k = line_search_tool.line_search(oracle, x_k, d_k)
        x_k += d_k * a_k

    df_last = oracle.grad(x_k)
    df_last_norm = df_last.dot(df_last)
    pushHistory(x_k, oracle, df_last_norm)
    if df_last_norm <= df0_norm * tolerance:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history


def subgradient_method(oracle, x_0, tolerance=1e-2, max_iter=1000, alpha_0=1,
                       display=False, trace=False):
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0) * 1.0
    f_k = oracle.func(x_k)
    minF_x = np.copy(f_k)
    minX_k = np.copy(x_k)
    alpha_k = lambda k: alpha_0 / (k + 1) ** (0.5)
    start = time.time()

    def pushHistory(x_k, f_k, gap):
        if trace:
            history['time'].append(time.time() - start)
            history['func'].append(f_k)
            history['duality_gap'].append(gap)
            if np.alen(x_k) <= 2:
                history['x'].append(np.copy(x_k))
        if display:
            print(x_k)

    for k in range(max_iter):
        if k > max_iter:
            return x_k, 'iterations_exceeded', history
        sub_df_k = oracle.subgrad(x_k)
        sub_df_k = sub_df_k / norm(sub_df_k)
        dual_gap = oracle.duality_gap(x_k)
        pushHistory(x_k, f_k, dual_gap)
        if dual_gap < tolerance:
            return minX_k, 'success', history
        x_k -= sub_df_k * alpha_k(k)
        f_k = oracle.func(x_k)
        if f_k < minF_x:
            minF_x = np.copy(f_k)
            minX_k = np.copy(x_k)

    dual_gap = oracle.duality_gap(x_k)
    pushHistory(x_k, f_k, dual_gap)
    if dual_gap < tolerance:
        return minX_k, 'success', history
    else:
        return minX_k, 'iterations_exceeded', history


def proximal_gradient_descent(oracle, x_0, L_0=1, tolerance=1e-5,
                              max_iter=1000, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_counter = 0
    x_k = np.copy(x_0) * 1.0
    L_k = L_0

    f_k = None
    df_k = None

    m_l = lambda y, x: f_k + df_k.dot(y - x) + L_k / 2 * (y - x).dot(y - x) + oracle._h.func(y)

    start = time.time()

    def pushHistory(x_k, oracle, gap):
        if trace:
            history['time'].append(time.time() - start)
            history['func'].append(oracle.func(x_k))
            history['duality_gap'].append(gap)
            history['line_counter'].append(np.copy(line_counter))
            if np.alen(x_k) <= 2:
                history['x'].append(np.copy(x_k))
        if display:
            print(x_k)

    for k in range(max_iter):
        if k > max_iter:
            return x_k, 'iterations_exceeded', history

        f_k = oracle._f.func(x_k)
        df_k = oracle.grad(x_k)

        dual_gap = oracle.duality_gap(x_k)
        pushHistory(x_k, oracle, dual_gap)
        if dual_gap < tolerance:
            return x_k, 'success', history

        while True:
            line_counter += 1
            y = oracle.prox(x_k - 1 / L_k * df_k, 1 / L_k)
            if oracle.func(y) <= m_l(y, x_k):
                break
            L_k *= 2
        x_k = np.copy(y)
        L_k = max(L_0, L_k / 2)

    dual_gap = oracle.duality_gap(x_k)
    pushHistory(x_k, oracle, dual_gap)
    if dual_gap < tolerance:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history


# def proximal_gradient_descent(oracle, x_0, L_0=1, tolerance=1e-5,
#                               max_iter=1000, trace=False, display=False):

def proximal_fast_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                                  max_iter=1000, trace=False, display=False):
    # TODO: Implement
    def fill_history():
        if not trace:
            return
        history['func'].append(func_min)
        history['time'].append((datetime.now() - t_0).seconds)
        history['duality_gap'].append(duality_gap_k)
        history['iterations'].append(iterations)
        if x_size <= 2:
            history['x'].append(np.copy(x_k))

    def do_display():
        if not display:
            return
        if x_size <= 4:
            print('x = {}, '.format(np.round(x_k, 4)), end='')
        print('func= {}, duality_gap = {}'.format(np.round(func_min, 10),
                                                  np.round(duality_gap_k, 4)))

    history = defaultdict(list) if trace else None
    x_k, L_k = np.copy(x_0), L_0
    A_k, v_k, y_k = 0, np.copy(x_k), np.copy(x_k)
    sum_diffs_k = 0
    iterations = 0
    t_0 = datetime.now()
    x_size = len(x_k)
    func_min = oracle.func(x_k)
    duality_gap_k = oracle.duality_gap(x_k)
    x_star = np.copy(x_k)

    for k in range(max_iter):
        if duality_gap_k < tolerance:
            break

        do_display()
        fill_history()

        while True:
            iterations += 1

            a_k = (1. + np.sqrt(1. + 4. * L_k * A_k)) / (2. * L_k)
            A_new = A_k + a_k
            y_k = (A_k * x_k + a_k * v_k) / A_new
            grad_k = oracle.grad(y_k)
            sum_diffs_new = sum_diffs_k + a_k * grad_k
            v_new = oracle.prox(x_0 - sum_diffs_new, A_new)
            x_new = (A_k * x_k + a_k * v_new) / A_new

            func_x = oracle.func(x_new)
            func_y = oracle.func(y_k)
            fs = np.array([func_min, func_x, func_y])
            func_min = np.min(fs)
            if func_min == func_x:
                x_star = x_new
            if func_min == func_y:
                x_star = y_k

            if oracle._f.func(x_new) > oracle._f.func(y_k) + grad_k @ (x_new - y_k) + \
                    L_k / 2 * np.linalg.norm(x_new - y_k) ** 2:
                L_k *= 2
            else:
                x_k = x_new
                A_k = A_new
                v_k = v_new
                sum_diffs_k = sum_diffs_new
                break

        L_k /= 2
        duality_gap_k = oracle.duality_gap(x_star)

    do_display()
    fill_history()
    message = 'success' if duality_gap_k < tolerance else 'iterations_exceeded'
    return x_star, message, history
