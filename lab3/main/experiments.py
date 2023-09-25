import optimization
import oracles
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(412)


def experiment1():
    for x_0, label in zip(
            [np.zeros(5), np.array([3, 5, 8, 2, 5]), np.array([3, 5, 8, 2, 5]) * 10, np.array([3, 5, 8, 2, 5]) * 100],
            ["0", "close", "medium", "far"]):
        plt.clf()
        res = []
        for a_0 in tqdm([a / 10 for a in range(1, 50)]):
            A = np.random.rand(5, 5)
            b = np.random.rand(5)
            r = 1 / 5
            oracle = oracles.create_lasso_nonsmooth_oracle(A, b, r)
            x_star, msg, hist = optimization.subgradient_method(oracle, x_0, alpha_0=a_0, trace=True, max_iter=10 ** 4)
            res.append(len(hist['func']))
        plt.plot([a / 10 for a in range(1, 50)], res)
        plt.xlabel("Alpha")
        plt.ylabel("Iterations")
        plt.show()
        # plt.savefig("exp1/{}.png".format(label))
        print('exp1 iter finished')


def experiment2():
    def plot_data(hist, n, lbl):
        param = "line_counter" if lbl == "gradient" else "iterations"
        plt.clf()
        plt.plot(range(len(hist[param])), hist[param], label=lbl, c='g')
        plt.plot(range(len(hist[param])), list(map(lambda x: 2 * x, range(len(hist[param])))),
                 label='2x', c='r')
        plt.legend()
        plt.xlabel("iters")
        plt.ylabel("line search attempts")
        plt.savefig("exp2/LCounter{}.png".format(n))
        plt.show()

    for n in tqdm([5, 50]):
        A = np.random.rand(n, n)
        b = np.random.rand(n)
        r = 1 / n

        oracle = oracles.create_lasso_prox_oracle(A, b, r)
        x_star, msg, hist = optimization.proximal_gradient_descent(oracle, np.zeros(n), trace=True)
        x_star1, msg1, hist1 = optimization.proximal_fast_gradient_method(oracle, np.zeros(n), trace=True)
        plot_data(hist, n, "gradient")
        plot_data(hist1, n, "fast_gradient")
        print('exp2 iter finished')


def experiment3():
    for n in tqdm([2 ** 6, 2 ** 8]):
        for r in [2 / 10, 4]:
            A = np.random.rand(n, n)
            b = np.random.rand(n)
            x_0 = np.zeros(n)
            hists = dict()
            for method, orac, lab in zip([optimization.subgradient_method, optimization.proximal_gradient_descent,
                                          optimization.proximal_fast_gradient_method],
                                         [oracles.create_lasso_nonsmooth_oracle, oracles.create_lasso_prox_oracle,
                                          oracles.create_lasso_prox_oracle],
                                         ["sub", "prox", "fast"]):
                oracle = orac(A, b, r)
                x_star, msg, hist = method(oracle, x_0, max_iter=10 ** 4, tolerance=10 ** -3, trace=True)
                hists[lab] = hist
            plt.clf()
            plt.figure(figsize=(16, 6))

            for lab, color in zip(["sub", "prox", "fast"], ["green", "blue", "red"]):
                plt.plot(range(len(hists[lab]['duality_gap'])), np.log(hists[lab]['duality_gap']), color=color,
                         label=lab)
            plt.xlabel("Iterations")
            plt.ylabel("Log of Gap")
            plt.title("iter-{}-{}".format(r, n))
            # plt.savefig("exp3/iter-{}-{}".format(r, n))
            plt.show()
            plt.clf()
            plt.figure(figsize=(16, 6))

            for lab, color in zip(["sub", "prox", "fast"], ["green", "blue", "red"]):
                plt.plot(hists[lab]['time'], np.log(hists[lab]['duality_gap']), color=color, label=lab)
            plt.xlabel("secs")
            plt.legend()
            plt.ylabel("Лог")
            plt.title("iter-{}-{}".format(r, n))
            # plt.savefig("exp3/second-{}-{}".format(r, n))
            plt.show()
            print('exp3 iter finished')
