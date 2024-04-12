
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    x = np.arange(-2, 2, 0.01)

    for i in range(20):

        sigma = 0.3
        eps = np.random.randn(1)*sigma

        y = (x-eps)**2
        plt.plot(x,y, color='blue', alpha=0.2)

    z = 1.0 / (2 * np.pi * sigma ** 2) ** 0.5 * np.exp(-(x ** 2) / (2 * sigma ** 2))

    lr = 0.01
    s = lr * 1 / (1-lr)
    s2 = 0.5 * 1 / (1-0.5)

    z2 = 1.0 / (2 * np.pi * s) ** 0.5 * np.exp(-(x ** 2) / (2 * s))

    z3 = 1.0 / (2 * np.pi * s2) ** 0.5 * np.exp(-(x ** 2) / (2 * s2))

    plt.title('stochastic quadratic model $f(x) = (x-\epsilon)^2$')
    plt.plot(x,z, color='red', label='noise distribution')
    plt.plot(x,z2, color='orange', label='equilibrium distribution with lr=0.01')
    plt.plot(x,z3, color='green', label='equilibrium distribution with lr=0.5')

    plt.legend()
    plt.show()


