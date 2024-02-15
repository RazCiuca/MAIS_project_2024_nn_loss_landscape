
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

def line_search_around_eigenvalues(model, data_x, data_y, loss_fn, eigvals, eigvecs, gradient):
    """
    in each eigendirection, choose n points around the minimum, say until you've increased the loss
    by some fixed factor, then partition the data into 512-sized chunks, then estimate the functional
    forms of all those chunks, in order to get statistics for the minimum locations
    """
    with t.no_grad():

        theta = eigvecs.T @ model.get_vectorized_params()
        theta_d = eigvecs.T @ gradient

        min_locs = theta - theta_d/eigvals

        n_params = theta.size(0)
        n_data = data_x.size(0)
        batch_indices = np.random.permutation(np.arange(n_data))
        batch_size = 512
        n_chunks = int(n_data/batch_size)
        n_explore = 20

        # where we store the function values
        f_values = t.zeros(n_params, n_chunks, n_explore)

        for i in range(n_chunks):
            indices = batch_indices[i*batch_size : (i+1)*batch_size]
            inputs = data_x[indices]
            targets = data_y[indices]

            for p in range(n_params):

                # todo: make sure this is right, I'm confused about tranposes here
                direction = eigvecs[p]

                for k in range(n_explore):

                    # todo: create the new params for this exploration point
                    z = model.shape_vec_as_params(z)
                    preds = functional_call(model, z, inputs)
                    loss = loss_fn(preds, targets)

                    f_values[p, i, k] = loss.item()


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

    for iters in [48800, 20000, 2000, 0]:

        model = ResNet9(3, 10, expand_factor=1)
        model.load_state_dict(t.load(f'models/resnet9_cifar10/model_{iters}.pth'))

        eigvals, eigvecs = t.load(f'./models/resnet9_cifar10/eig_{iters}/eigvals_vecs.pth')
        gradient = t.load(f'./models/resnet9_cifar10/gradients/{iters}.pth')

        # model = model.cuda()
        loss_fn = nn.CrossEntropyLoss()

        f_values = line_search_around_eigenvalues(model, data_x, data_y, loss_fn, eigvals, eigvecs, gradient)
        t.save(gradient, f'./models/resnet9_cifar10/gradients/{iters}.pth')


