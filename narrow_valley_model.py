"""
In this file we test properties of the narrow valley model, and see what it can explain
about the eigenspectrum

"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch as t
from scipy.stats import ortho_group
from numpy.linalg import eigh

def f_narrow_valley(x, x_0, H, beta):
    y = x - x_0
    factor = t.exp(beta @ y)
    Hy = H @ y
    f = 0.5 * (y @ Hy) * factor

    return f

def f_narrow_valley_linear(x, x_0, H, beta):

    y = x - x_0
    factor = 1 + beta @ y
    Hy = H @ y
    yHy = 0.5 * y @ Hy
    f = yHy * factor

    return f

def narrow_valley_compute(x, x_0, H, beta):

    y = x - x_0

    factor = t.exp(beta @ y)

    Hy = H @ y

    f = 0.5 * (y @ Hy) * factor

    grad_f = factor * (Hy) + f * beta

    sym_term = grad_f.unsqueeze(1) @ beta.unsqueeze(0) + beta.unsqueeze(1) @ grad_f.unsqueeze(0)
    grad2_f = factor * H + sym_term - f * beta.unsqueeze(1) @ beta.unsqueeze(0)

    return f, grad_f, grad2_f


def f_narrow_valley_generic(x, x_0, H, beta, sigma):

    y = x - x_0
    z = beta @ y
    f0 = 1 + sigma(z)
    Hy = H @ y
    yHy = y @ Hy
    f_val = 0.5 * yHy * f0

    return f_val

def narrow_valley_compute_generic(x, x_0, H, beta, sigma, sigma_p, sigma_pp):

    y = x - x_0
    z = beta @ y

    f0 = 1 + sigma(z)
    f1 = sigma_p(z)
    f2 = sigma_pp(z)

    Hy = H @ y

    yHy = 0.5 * y @ Hy

    f_val = yHy * f0

    grad_f = f0 * (Hy) + (y @ Hy) * f1 * beta

    sym_term = f1 * (Hy.unsqueeze(1) @ beta.unsqueeze(0) + beta.unsqueeze(1) @ Hy.unsqueeze(0))
    grad2_f = f0 * H + sym_term + yHy * f2 * (beta.unsqueeze(1) @ beta.unsqueeze(0))

    return f_val, grad_f, grad2_f

def narrow_valley_compute_linear(x, x_0, H, beta):

    y = x - x_0

    factor = 1 + beta @ y

    Hy = H @ y

    yHy = y @ Hy

    f = 0.5 * yHy * factor

    grad_f = factor * (Hy) + 0.5 * yHy * beta

    sym_term = Hy.unsqueeze(1) @ beta.unsqueeze(0) + beta.unsqueeze(1) @ Hy.unsqueeze(0)
    grad2_f = (factor) * H + sym_term

    return f, grad_f, grad2_f


if __name__ == "__main__":

    dim = 20000

    # generate orthogonal matrix
    # Q = t.from_numpy(ortho_group.rvs(dim=dim)).float()

    # ========================================================================
    # These now work, don't change without cause
    # ========================================================================

    # generate power law positive spectrum and initial Hessian
    eigen = 100/(t.arange(1, dim+1)**1)
    H0 = t.diag_embed(eigen)

    # weighing = 1.0/eigen
    x = t.randn(dim)
    # compute hessian of model and new spectrum
    beta = t.randn(dim)*20

    x = x - (x@beta + 3) * beta / (beta.norm()**2)

    # ========================================================================
    # These now work, don't change without cause
    # ========================================================================

    print(x@beta)

    # beta = t.arange(1, dim+1).float()
    # beta = beta * dim**0.5 / beta.norm()
    x0 = t.zeros(dim)

    k = 1
    sigma = lambda x: t.exp(k*x)
    sigma_p = lambda x: k * t.exp(x)
    sigma_pp = lambda x: k * k * t.exp(x)

    # ============================================
    # Do SGD on the function to find the equilibrium
    # ============================================
    variance = 1e-1
    x.requires_grad = True

    optimizer = t.optim.SGD([x], lr=5e-1, momentum=0)

    for i in range(0, 0):
        optimizer.zero_grad()
        noise = t.randn(dim) * variance**0.5
        # noise -= (noise @ beta) * beta / (beta.norm()**2)
        loss = f_narrow_valley_generic(x, x0, H0, beta, sigma=sigma)

        loss_total = (x - x0 ) @ (H0 @ noise) + loss

        loss_total.backward()

        # check if gradient is correct:
        # f, grad_f, grad2_f = narrow_valley_compute(x, x0, H0, beta)
        #
        # grad_diff = (x.grad - grad_f).norm()

        optimizer.step()
        # print(f"step {i} loss:{loss.item():.5f} grad diff:{grad_diff.item()} ")
        print(f"step {i} loss:{loss.item():.5e}")

    x = x.detach()

    # ============================================
    # Plotting
    # ============================================
    f, grad_f, grad2_f = narrow_valley_compute_generic(x, x0, H0, beta, sigma=sigma, sigma_p=sigma_p, sigma_pp=sigma_pp)

    eigvals, eigvecs = t.linalg.eigh(grad2_f)
    mask = eigvals > 0

    print(f"n negative values:{(eigvals<0).sum().item()}")

    plt.subplot(2,1,1)
    # plt.plot(eigen.numpy(), label='original power law')
    plt.plot(eigvals[mask].numpy()[::-1], label='new spectrum')
    plt.plot(2*eigen, label='initial spectrum')

    plt.legend()
    plt.yscale('log')
    plt.xscale('log')

    plt.subplot(2,1,2)
    plt.plot(np.abs(eigvals[eigvals<0].numpy()))

    plt.yscale('log')
    plt.xscale('log')
    plt.show()

    # ============================================
    # Line search in most negative eigenvalue
    # ============================================

    # for i in range(0, 3):
    #
    #     search_dir = eigvecs[:, i]
    #
    #     alpha = t.arange(-1, 1, 1e-2)
    #
    #     f_values = np.array([f_narrow_valley(x + a*search_dir, x0, H0, beta) for a in alpha])
    #
    #     plt.plot(f_values)
    #
    # plt.show()

