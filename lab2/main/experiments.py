import oracles
import optimization

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_svmlight_file
from datetime import datetime
from scipy.sparse import diags
from tqdm import tqdm

np.random.seed(seed=42)


def experiment_1():
    mus = [10, 100, 1000, 10000]
    iters = 2 ** 4
    ks = np.arange(1, 1000, 100)
    results = {}

    vars = zip(mus, ['y', 'b', 'g', 'r'])

    for n, color in vars:
        # results
        results[n] = [[] for _ in range(iters)]
        for i in range(iters):
            for k in ks:

                M = np.random.uniform(low=1, high=k, size=n)
                M[0] = 1
                M[-1] = k
                A = diags(M)
                b = np.random.uniform(low=1, high=k, size=n)

                matvec = lambda x: A @ x

                x_star, msg, history = optimization.conjugate_gradients(matvec, b, np.zeros(n),
                                                                        trace=True)
                if msg == 'success':
                    results[n][i].append(len(history['time']))
            plt.plot(ks, results[n][i], ls='--', color=color)
        plt.plot(ks, np.mean(results[n], axis=0), color=color, label='n = {}'.format(n))

    plt.grid()
    plt.legend()
    plt.xlabel('mus')
    plt.ylabel('iters')
    plt.savefig('./experiments/experiment_1')


def experiment_2():
    # define paths
    path = './data/gisette_scale'

    # load data
    A, b = load_svmlight_file(path)
    m, n = A.shape

    oracle = oracles.create_log_reg_oracle(A, b, 1 / m)
    results = []

    ls = [0, 1, 5, 10, 50, 100]
    for l in tqdm(ls):
        _, _, hist = optimization.lbfgs(oracle, np.zeros(n), memory_size=l, trace=True)

        norm = np.array(hist['grad_norm'])
        norm /= norm[0]
        norm = np.log(norm ** 2)

        results.append((l, norm, hist['time']))

    # plotting iters
    for l, grad_norm, times in results:
        plt.plot(list(range(len(grad_norm))), grad_norm, label=f'|history| {l}')
    plt.figure(figsize=(12, 8))
    plt.xlabel('iters')
    plt.ylabel(r'grad_norm')
    plt.legend()
    plt.grid()
    plt.plot()
    plt.savefig('./experiments/grad_norm-iters')
    plt.cla()

    # plotting seconds
    for l, grad_norm, times in results:
        plt.plot(times, grad_norm, label='history size = {}'.format(l))
    plt.figure(figsize=(12, 8))
    plt.xlabel('secs')
    plt.ylabel(r'grad_norm')
    plt.legend()
    plt.grid()
    plt.plot()
    plt.savefig('./experiments/grad_norm-seconds')
    plt.cla()


def experiment_3():
    paths = [
        'gisette_scale',
        'news20.binary',
        'real-sim',
        'w8a'
        'rcv1_train.binary',
    ]

    optimization_methods = [
        optimization.hessian_free_newton,
        optimization.lbfgs,
        optimization.gradient_descent
    ]

    def plotting(hfn_history, lbfgs_history, gd_history, dataset):

        histories = [hfn_history, lbfgs_history, gd_history]
        colors = ['b', 'r', 'g']
        names = ['hessian_free_newton', 'lbfgs', 'gradient_descent']
        for x_value in ['iterations', 'time']:
            for y_value in ['func', 'grad']:
                if (x_value, y_value) == ('iterations', 'grad'):
                    continue
                plt.figure()
                for history, name, color in zip(histories, names, colors):

                    if x_value == 'iterations':
                        x1 = list(range(len(history['time'])))
                    else:
                        x1 = history['time']

                    if y_value == 'func':
                        x2 = history['func']
                    else:
                        norm = np.array(hist['grad_norm'])
                        norm /= norm[0]
                        norm = np.log(norm ** 2)
                        x2 = norm

                    plt.plot(x1, x2, label=name, color=color)

                # set params
                plt.title(dataset)
                plt.xlabel(x_value)
                if y_value == 'func':
                    plt.ylabel(y_value)
                else:
                    plt.ylabel(r'$\log\left(grad\_norm\right)$')
                plt.grid()
                plt.legend()
                plt.savefig(f'experiment_3/{dataset}_{x_value}-vs-{y_value}.jpg')
                plt.plot()

    for dataset in tqdm(paths):

        A, b = load_svmlight_file(f'./data/{dataset}')
        n, m = A.shape
        oracle = oracles.create_log_reg_oracle(A, b, 1 / n)
        hists = []

        for method in optimization_methods:
            _, _, hist = method(oracle, np.zeros(m), trace=True)
            hists.append(hist)

        plotting(*hists, dataset)
