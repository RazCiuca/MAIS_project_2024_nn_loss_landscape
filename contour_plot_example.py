"""
makes contour plot (and other plots) of the narrowing valley function for illustration purposes

"""
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import matplotlib.cm as cm
from scipy.optimize import root, approx_fprime
import matplotlib.ticker as ticker
from numpy.linalg import eigh

def fn_dist(x,y,t):
    return (x-t)**2 + (y - t+t**2)**2

def get_min_t(x,y):
    res = root(fun=(lambda t : (-2*(x-t) + 2*(y-t+t**2) * (-1+2*t))), x0=np.array([0]))
    return res.x

# def func_to_plot(x,y):
#
#     # find minimum t:
#     t = get_min_t(x, y)
#     dist_squared = fn_dist(x,y,t)
#
#     # return 5*np.log(dist_squared+1) * (-1/(3*(np.abs(t)-1.1))) + 0.2 * (t-0.8)**2
#     return 10*dist_squared * (-1/(3*(np.abs(t)-1.1))) + 0.2 * (t-0.8)**2

def func_to_plot(x,y):

    # return 5*np.log(dist_squared+1) * (-1/(3*(np.abs(t)-1.1))) + 0.2 * (t-0.8)**2
    # return np.log(0.2*(x-0.8)**2 + y**2 * np.exp(5*(x)+1) + 1)
    # return np.log((1+np.exp(x-5))*(0.1 * (x - 5) ** 2 + 5*y ** 2)  + 1)
    return (np.exp((x-5)/5))*(0.04 * (x - 5) ** 2 + 5*y ** 2)

    # return np.log(0.2 * (x - 0.8) ** 2 + y ** 2 * (1+20*x) + 1)

def find_gradient(x,y):

    z = [x,y]

    eps = 1e-3
    base = func_to_plot(z[0], z[1])
    dx = func_to_plot(z[0] + eps, z[1]) - base
    dy = func_to_plot(z[0], z[1] + eps) - base
    return np.array([dx / eps, dy / eps])

def find_hessian_eigvecs(x,y):
    """
    finds the hessian eigenvectors and values at a given point
    """

    # approx_fprime to find the derivatives
    def grad_fn(z):
        eps = 1e-3
        base = func_to_plot(z[0], z[1])
        dx = func_to_plot(z[0] + eps, z[1]) - base
        dy = func_to_plot(z[0], z[1] + eps) - base
        return np.array([dx/eps, dy/eps])

    eps = 1e-3
    grad_base = grad_fn([x,y])
    grad_dx = grad_fn([x + eps, y])
    grad_dy = grad_fn([x, y + eps])

    hessian = np.array([(grad_dx-grad_base)/eps, (grad_dy-grad_base)/eps]).squeeze()

    eigvals, eigvecs = eigh(hessian)

    return eigvals, eigvecs

if __name__ == "__main___":

    hessian = find_hessian_eigvecs(0.5, 0.2)



if __name__ == "__main__":

    delta = 0.02
    x = np.arange(0, 6, delta)
    y = np.arange(-4, 4, delta)
    X, Y = np.meshgrid(x, y)

    Z = np.array([func_to_plot(x,y) for x,y in zip(X.flatten(), Y.flatten())]).reshape(X.shape)

    print(Z.shape)

    # ====================================================
    # plotting Contour lines
    # ====================================================

    fig, ax = plt.subplots()
    cs = ax.contour(X, Y, Z, levels=40)

    # x2 = np.arange(0, 1, 0.01)
    # y2 = x2 - x2**2
    # ax.plot(x2, y2, label='valley')
    ax.scatter(np.array([5]), np.array([0]), color='red', label='minimum')

    # ====================================================
    # plotting gradients on graph
    # ====================================================
    # x, y = np.meshgrid(np.arange(0, 1.0, 0.02), np.arange(-0.2, 0.2, 0.02))
    # gradients = np.array([find_gradient(x, y) for x, y in zip(x.flatten(), y.flatten())])
    # gradients = gradients.squeeze() / (np.sqrt(gradients[:, 0]**2 + gradients[:, 1]**2)).reshape(-1, 1)
    # u = np.array([-x[0] for x in gradients]).reshape(x.shape)
    # v = np.array([-x[1] for x in gradients]).reshape(x.shape)
    # ax.quiver(x, y, u, v, alpha=0.6)

    # ====================================================
    # plotting where grad_x is negative
    # ====================================================
    # x, y = np.meshgrid(np.arange(0, 6.0, 0.01), np.arange(-4, 4, 0.01))
    # gradients = np.array([find_gradient(x, y) for x, y in zip(x.flatten(), y.flatten())])
    # neg_grad_x = np.array([(x[0] < 0) for x in gradients]).reshape(x.shape)
    # cs2 = ax.contourf(x, y, neg_grad_x, cmap='coolwarm', alpha=0.7, label='negative x gradient')

    # ====================================================
    # plotting the eigenvectors on the graph
    # ====================================================
    # locations of quiver points
    # x, y = np.meshgrid(np.arange(0, 1.1, 0.02), np.arange(0, 0.5, 0.02))
    #
    # # for all points in grid, find the eigenvectors, and define C map via the eigenvalues
    # eigenstuff = [find_hessian_eigvecs(x, y) for x, y in zip(x.flatten(), y.flatten())]
    # eigenvals = [x[0] for x in eigenstuff]
    # eigenvecs = [x[1] for x in eigenstuff]
    #
    # # vectors and values 1:
    # c1 = np.array([  2*(x[0]>0)-1    for x in eigenvals]).reshape(x.shape)
    # u1 = np.array([x[0, 0] for x in eigenvecs]).reshape(x.shape)
    # v1 = np.array([x[1, 0] for x in eigenvecs]).reshape(x.shape)
    #
    # # vectors and values 2
    # c2 = np.array([2*(x[1]>0)-1  for x in eigenvals]).reshape(x.shape)
    # u2 = np.array([x[0, 1] for x in eigenvecs]).reshape(x.shape)
    # v2 = np.array([x[1, 1] for x in eigenvecs]).reshape(x.shape)
    #
    # ax.quiver(x, y, u1, v1, c1, alpha=0.6)
    # ax.quiver(x, y, u2, v2, c2, alpha=0.6)

    # ====================================================
    # plotting a contourf plot of where there are negative eigenvalues
    # ====================================================
    X, Y = np.meshgrid(np.arange(0, 6.0, 0.05), np.arange(-4, 4, 0.05))
    eigenstuff = [find_hessian_eigvecs(x, y) for x, y in zip(X.flatten(), Y.flatten())]
    neg_eigen_locs = np.array([np.sum(x[0] < 0)*2-1 for x in eigenstuff]).reshape(X.shape)
    max_eigen = np.array([x[0].max() for x in eigenstuff]).reshape(X.shape)

    cs2 = ax.contourf(X, Y, neg_eigen_locs, cmap='coolwarm', alpha=0.6, label='negative eigenvalues')
    # cs2 = ax.contourf(X, Y, max_eigen, cmap='coolwarm', alpha=0.6, levels=20)

    ax.legend()
    # ax.label('red regions have negative eigenvalues, blue regions don\'t')
    # ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title('Long narrow valley with tightening walls')
    cbar = fig.colorbar(cs)
    fig.show()


