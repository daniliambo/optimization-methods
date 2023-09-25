from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

import numpy as np

from utils import get_line_search_tool


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)

    def fill_history():
        history['time'].append((datetime.now() - t0).seconds)
        history['residual_norm'].append(norm_grad_k)
        if len(x_k) <= 2:
            history['x'].append(np.copy(x_k))

    def show():
        print('x: {}, norm: {}'.format(x_k, norm_grad_k))

    message = None
    g_k = matvec(x_k) - b
    norm_grad_k = np.linalg.norm(g_k)
    d_k = -g_k

    t0 = datetime.now()

    # define max_iter
    if not max_iter:
        max_iter = 2 * len(x_k)
    else:
        max_iter = min(max_iter, 2 * len(x_k))

    for _ in tqdm(range(max_iter)):
        show()
        fill_history()

        adk = matvec(d_k)
        alpha = (g_k.T @ g_k) / (d_k.T @ adk)
        x_k = x_k + alpha * d_k
        prevgk = np.copy(g_k)
        g_k = g_k + alpha * adk
        norm_grad_k = np.linalg.norm(g_k)
        if norm_grad_k <= tolerance * np.linalg.norm(b):
            message = 'success'
            break

        # set dir
        b = (g_k.T @ g_k) / (prevgk.T @ prevgk)
        d_k = -g_k + b * d_k

    if trace:
        fill_history()
    if display:
        show()

    if not norm_grad_k <= tolerance * np.linalg.norm(b):
        message = 'iterations_exceeded'

    return x_k, message, history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # TODO: Implement L-BFGS method.
    def fill_history():
        history['func'].append(oracle.func(x_k))
        history['time'].append((datetime.now() - t_0).seconds)
        history['grad_norm'].append(grad_k_norm)
        if x_size <= 2:
            history['x'].append(np.copy(x_k))

    def show():
        if len(x_k) <= 4:
            print('x = {}, '.format(np.round(x_k, 4)), end='')
        print('func= {}, grad_norm = {}'.format(np.round(oracle.func(x_k), 4),
                                                np.round(grad_k_norm, 4)))

    t_0 = datetime.now()
    x_size = len(x_k)
    message = None

    grad_k = oracle.grad(x_k)
    grad_0_norm = grad_k_norm = np.linalg.norm(grad_k)

    def bfgs_multiply(v, H, gamma_0):
        if len(H) == 0:
            return gamma_0 * v
        s, y = H[-1]
        H = H[:-1]
        v_new = v - (s @ v) / (y @ s) * y
        z = bfgs_multiply(v_new, H, gamma_0)
        result = z + (s @ v - y @ z) / (y @ s) * s
        return result

    def bfgs_direction():
        if len(H) == 0:
            return -grad_k
        s, y = H[-1]
        gamma_0 = (y @ s) / (y @ y)
        return bfgs_multiply(-grad_k, H, gamma_0)

    H = []
    for _ in tqdm(range(max_iter)):
        show()
        fill_history()

        d = bfgs_direction()
        alpha = line_search_tool.line_search(oracle, x_k, d)
        x_new = x_k + alpha * d
        grad_new = oracle.grad(x_new)
        H.append((x_new - x_k, grad_new - grad_k))
        if len(H) > memory_size:
            H = H[1:]
        x_k, grad_k = x_new, grad_new
        grad_k_norm = np.linalg.norm(grad_k)
        if grad_k_norm ** 2 < tolerance * grad_0_norm ** 2:
            message = 'success'
            break

    if trace:
        fill_history()
    if display:
        show()

    if not grad_k_norm ** 2 < tolerance * grad_0_norm ** 2:
        message = 'iterations_exceeded'

    return x_k, message, history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500,
                        line_search_options=None, display=False, trace=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    def fill_history():
        history['func'].append(oracle.func(x_k))
        history['time'].append((datetime.now() - t0).seconds)
        history['grad_norm'].append(grad_k_norm)
        if x_size <= 2:
            history['x'].append(np.copy(x_k))

    def show():
        if len(x_k) <= 4:
            print('x = {}, '.format(np.round(x_k, 4)), end='')
        print('func= {}, grad_norm = {}'.format(np.round(oracle.func(x_k), 4),
                                                np.round(grad_k_norm, 4)))

    t0 = datetime.now()
    x_size = len(x_k)
    message = None

    grad_k = oracle.grad(x_k)
    grad_0_norm = grad_k_norm = np.linalg.norm(grad_k)

    for _ in tqdm(range(max_iter)):
        show()
        fill_history()

        eps = min(0.5, grad_k_norm ** 0.5)
        while True:
            hess_vec = lambda v: oracle.hess_vec(x_k, v)
            d, _, _ = conjugate_gradients(hess_vec, -grad_k, -grad_k, eps)
            if grad_k @ d < 0:
                break
            else:
                eps *= 10
        alpha = line_search_tool.line_search(oracle, x_k, d, previous_alpha=1)
        x_k = x_k + alpha * d
        grad_k = oracle.grad(x_k)
        grad_k_norm = np.linalg.norm(grad_k)
        if grad_k_norm ** 2 < tolerance * grad_0_norm ** 2:
            message = 'success'
            break

    if trace:
        fill_history()
    if display:
        show()

    if not grad_k_norm ** 2 < tolerance * grad_0_norm ** 2:
        message = 'iterations_exceeded'

    return x_k, message, history


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    def fill_history():
        if not trace:
            return
        history['time'].append((datetime.now() - t_0).seconds)
        history['func'].append(func_k)
        history['grad_norm'].append(grad_k_norm)
        if len(x_k) <= 2:
            history['x'].append(np.copy(x_k))

    t_0 = datetime.now()
    func_k = oracle.func(x_k)
    grad_k = oracle.grad(x_k)
    a_k = None
    grad_0_norm = grad_k_norm = np.linalg.norm(grad_k)
    fill_history()
    if display:
        print('New GD')

    for i in tqdm(range(max_iter)):
        if display:
            print('i: {}, norm: {}, func: {}, x: {}, grad: {}'.format(i,
                                                                      grad_k_norm,
                                                                      func_k,
                                                                      x_k, grad_k), end=' ')
        if grad_k_norm ** 2 <= tolerance * grad_0_norm ** 2:
            break

        d_k = -grad_k
        a_k = line_search_tool.line_search(oracle, x_k, d_k, 2 * a_k if a_k else None)
        if display:
            print('alpha: {}'.format(a_k))
        x_k += a_k * d_k
        func_k = oracle.func(x_k)
        grad_k = oracle.grad(x_k)
        grad_k_norm = np.linalg.norm(grad_k)
        fill_history()
    if display:
        print()

    if grad_k_norm ** 2 <= tolerance * grad_0_norm ** 2:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history
