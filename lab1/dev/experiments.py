import inspect

import scipy.sparse

import optimization, oracles
import numpy as np
import matplotlib.pyplot as plt
import plot_trajectory_2d as plot
from tqdm import tqdm


class Experiment:
    def __init__(self):
        pass

    def generate_data(self, mu, n, matrix_creation=None):
        if matrix_creation["hard_mode"]:
            a = np.random.randn(n, n)
            A = (a + a.T) / 2
            x = np.random.randint(0, mu, size=n)
            x[0], x[-1] = 1, mu
            np.fill_diagonal(A, x)
        elif matrix_creation["sparse"]:
            # a = np.random.choice(np.arange(1, mu), size=n, replace=True)
            # A = scipy.sparse.diags(a)
            a = np.random.choice(np.arange(1, mu), n, replace=True)
            a[0] = 1
            a[-1] = mu
            A = scipy.sparse.diags(a)
        else:
            A = np.eye(n)
            x = np.random.randint(0, mu, size=n)
            x[0], x[-1] = 1, mu
            np.fill_diagonal(A, x)

        return A, np.random.normal(size=(n))

    def getOracles(self, mu, n, matrix_creation=None):
        A, b = self.generate_data(mu, n, matrix_creation)

        # oracle 1
        oracle1 = oracles.QuadraticOracle(A, b)

        # oracle 2
        oracle2 = oracles.QuadraticOracle(A, b)

        oracle2.func = lambda x: 1 / 2 * np.linalg.norm(A @ x - b) ** 2
        oracle2.grad = lambda x: A.T @ (A @ x - b)
        oracle2.hess = lambda x: A.T @ A

        return oracle1, oracle2

    def perform_optimization(self, params, optimization_method):
        oracle = params["oracle"]
        method = params["method"]
        kwargs = params["kwargs"]
        starting_point = params["starting_point"]

        line_search_tool = optimization.LineSearchTool(method, **kwargs[method])
        if optimization_method == "gradient_descent":
            x_k, status, history = optimization.gradient_descent(oracle, starting_point,
                                                                 line_search_options=None,
                                                                 trace=True,
                                                                 display=False)
        else:
            x_k, status, history = optimization.newton(oracle, starting_point,
                                                       line_search_options=line_search_tool,
                                                       trace=True,
                                                       display=False)
        return x_k, status, history

    def fill_params(self, oracle, method, starting_point):
        kwargs = {"Wolfe": {"c1": 1e-4, "c2": 0.5, "alpha_0": 1.0}, "Armijo": {"c1": 1e-4, "alpha_0": 1.0},
                  "Constant": {"c": 0.3}}
        params = {"oracle": oracle, "method": method, "kwargs": kwargs, "starting_point": starting_point}
        return params

    def experiment(self, oracle, mu, method, start, optimization_method, plotting=True):
        params = self.fill_params(oracle, method, start)
        x_k, status, history = self.perform_optimization(params, optimization_method)
        if plotting:
            func = inspect.getsource(oracle.func)
            print(
                "with func: {}\nwith optimization method: {}\nwith method: {}\nwith mu: {}\nwith starting point: {}\niterations until convergence: {}"
                .format(func.splitlines()[-1][8:], optimization_method, method, mu, start, len(history["func"])))

            oracle = params["oracle"]
            plt.xlabel('x1')
            plt.ylabel('x2')
            plot.plot_levels(oracle.func)
            plot.plot_trajectory(oracle.func, history["x"])
            plt.savefig(f'{oracle}.png')
            plt.show()

        return len(history["func"])

    def calculate_epoch_mu(self, mus, method, start, n=2, optimization_method="gradient_descent",
                           matrix_creation=None):
        iters = []
        for mu in mus:
            oracle1, oracle2 = self.getOracles(mu, n, matrix_creation)
            iter = self.experiment(oracle1, mu, method, start, optimization_method, plotting=False)
            iters = np.append(iters, iter)

        # get rid of divergent iters
        iters = np.where(iters < 10000, iters, 0)
        return iters

    def calculate_epoch_start(self, mu, method, starts, optimization_method="gradient_descent", matrix_creation=None):
        n = 2
        iters = []
        for start in starts:
            oracle1, oracle2 = self.getOracles(mu, n, matrix_creation)
            iter = self.experiment(oracle1, mu, method, start, optimization_method, plotting=False)
            iters = np.append(iters, iter)

        # get rid of divergent iters
        iters = np.where(iters < 10000, iters, 0)
        return iters

    def calculate_epoch_method(self, methods, mus, start, optimization_method="gradient_descent", matrix_creation=None):
        n = 2
        v = [[]] * len(methods)
        d = dict(zip(methods, v))

        for method in tqdm(methods):
            iters = []
            for mu in mus:
                oracle1, oracle2 = self.getOracles(mu, n, matrix_creation)
                iter = self.experiment(oracle1, mu, method, start, optimization_method, plotting=False)
                iters = np.append(iters, iter)
            iters = np.where(iters < 10000, iters, 0)
            d[method] = iters

        return d

    def plot_param(self, epoch_avg, label, array, method=False):
        # if method:
        plt.figure(figsize=(16, 8))
        plt.grid()
        if method:
            plt.title(method)
        plt.xlabel(label)
        plt.ylabel("iters")
        plt.xticks(np.arange(len(array)), array)
        plt.plot(epoch_avg.astype(int))
        plt.show()

    def plot_param_2(self, d, mus, ns, cmap):
        cmap = cmap[:len(ns)]
        plt.figure(figsize=(16, 8))

        plt.xlabel("mus")
        plt.ylabel("iters")
        plt.xticks(np.arange(len(mus)), mus)
        plt.grid()

        for i, n in enumerate(ns):
            plt.plot(np.array(d[n][:-1]).T, linestyle='dotted', color=cmap[i])
            plt.plot(d[n][-1], label=f"n = {n}", color=cmap[i], linestyle="solid", linewidth=5)

        plt.legend()
        plt.show()
