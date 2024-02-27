
import time
import torch as t
import torch.optim as optim
import torchvision
from torch.func import functional_call
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, eigsh

from torch.func import jvp, grad, vjp
from torch.autograd.functional import vhp
from models import *

def min_variance_around_eigenvalues(model, data_x, data_y, loss_fn, eigvals, eigvecs, gradient, eigindices, n_explore=None):
    """
    in each eigendirection, choose n points around the minimum, say until you've increased the loss
    by some fixed factor, then partition the data into 512-sized chunks, then estimate the functional
    forms of all those chunks, in order to get statistics for the minimum locations
    """
    model_params = model.get_vectorized_params()

    with t.no_grad():

        theta = eigvecs.T @ model_params
        theta_d = eigvecs.T @ gradient

        min_locs = - theta_d/eigvals

        n_params = theta.size(0)
        n_data = data_x.size(0)
        batch_indices = np.random.permutation(np.arange(n_data))
        batch_size = 512
        n_chunks = int(n_data/batch_size)
        n_explore = 20 if n_explore is None else n_explore

        # where we store the function values
        f_values = t.zeros(len(eigindices), n_chunks, n_explore)

        largest_eigval = eigvals[-1]

        start = time.time()
        for i in range(n_chunks):

            stop = time.time()

            time_remaining = (stop - start) * ((n_chunks - i) / (1 if i==0 else i))
            print(f"doing chunk {i}/{n_chunks}, time remaining:{time_remaining}")
            indices = batch_indices[i*batch_size : (i+1)*batch_size]
            inputs = data_x[indices]
            targets = data_y[indices]

            params_to_search = eigindices

            for index, p in enumerate(params_to_search):

                direction = eigvecs[:, p]
                dir_eigval = eigvals[p]

                z = (0.1/dir_eigval)**0.5
                Xs = np.arange(-z, z, 2*z/n_explore)
                Ys = []

                for k,coef in enumerate(Xs):

                    z = model_params + coef * direction

                    z = model.shape_vec_as_params(z)
                    preds = functional_call(model, z, inputs)
                    loss = loss_fn(preds, targets)

                    f_values[index, i, k] = loss.item()
                    Ys.append(loss.item())

                # todo: interpolate using CubicSpline
                # todo: find minima with scipy
                # todo: find variance of minima and plot with respect to eigenvalue

                # y = f_values[p2, i].numpy()

                # print(y - y.min())

        return f_values

