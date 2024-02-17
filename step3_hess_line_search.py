"""
todo: use scipy.interpolation to much more efficiently find the minimum, then compute
a small number of points around the minimum, and find the relationship between minimum variance
and eigenvalue

prediction: find the expected loss decrease to be found by exactly optimising all the eigenvalues

baffling observation right now: how do we explain the change in spectrum throughout optimisation?
second question: if minima in eigendirections don't stay minima, how do we contextualize the
optimisation?

if the given eigendirections are very well approximated by quadratics, it doesn't make sense that
the spectrum would be shifted upwards after more optimisation.

Perhaps more optimisaiton would do that, but certainly not in the initial, lr decrease stage

prediction of the model: the initial decrease to a lower noise level does not much change the spectrum.
So, the spectrum between the 10000 and 12000 points should not be much different

if I go to the minimum of all known positive eigenvalues, and then again do line searches in
all known directions, will the spectrum change?

task: create a toy model which also changes spectrum when we go to a lower level.

todo: estimate how much of the initial drop in loss is atributable to a noise decrease, and how much
      is attributable to settling into newer crevasses

"""


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

def line_search_around_eigenvalues(model, data_x, data_y, loss_fn, eigvals, eigvecs, gradient, eigindices, n_explore=None):
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

                for k in range(n_explore):

                    coef = 0.1 * (2*(k+1)/n_explore - 1)

                    z = model_params + coef * direction

                    z = model.shape_vec_as_params(z)
                    preds = functional_call(model, z, inputs)
                    loss = loss_fn(preds, targets)

                    f_values[index, i, k] = loss.item()

                # y = f_values[p2, i].numpy()

                # print(y - y.min())

        return f_values


# line search around eigenvalues
if __name__ == "__main__":

    # data_x, data_y = t.load('models/resnet9_cifar10/enlarged_dataset.pth')

    data_train = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True)
    data_x = t.from_numpy(data_train.data).float()
    data_y = t.LongTensor(data_train.targets)

    x_mean = data_x.mean(dim=0)
    x_std = data_x.std(dim=0)

    data_x = (data_x - x_mean) / (1e-7 + x_std)
    data_x = data_x.transpose(1, 3)

    # for iters in [48800, 20000, 2000, 0]:
    for iters in [0]:

        model = ResNet9(3, 10, expand_factor=1)
        model.load_state_dict(t.load(f'models/resnet9_cifar10/model_{iters}.pth'))

        eigvals, eigvecs = t.load(f'./models/resnet9_cifar10/eig_{iters}/eigvals_vecs.pth')
        gradient = t.load(f'./models/resnet9_cifar10/gradients/{iters}.pth')

        # model = model.cuda()
        loss_fn = nn.CrossEntropyLoss()

        n_params = eigvals.size(0)

        eigindices = list(n_params - np.exp(np.arange(0, np.log(n_params), np.log(n_params)/30)).astype(int))

        for i in range(len(eigindices)-1):
            if eigindices[i+1] in eigindices[:i+1]:
                eigindices[i+1] = min(eigindices[:i+1]) - 1

        eigindices.append(0)

        print(eigindices)
        plotted_eigvals = eigvals[eigindices]

        f_values = line_search_around_eigenvalues(model, data_x, data_y, loss_fn, eigvals, eigvecs, gradient,
                                                  eigindices=eigindices, n_explore=40)
        t.save(f_values, f'./models/resnet9_cifar10/line_searches/{iters}_top_10_largest.pth')


