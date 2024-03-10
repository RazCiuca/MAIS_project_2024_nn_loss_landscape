


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

def line_search_around_eigenvalues(model, data_x, data_y, loss_fn, eigvals, eigvecs, gradient, eigindices=None, n_explore=None):
    """
    in each eigendirection, choose n points around the minimum, say until you've increased the loss
    by some fixed factor, then partition the data into 512-sized chunks, then estimate the functional
    forms of all those chunks, in order to get statistics for the minimum locations
    """
    model_params = model.get_vectorized_params()

    with t.no_grad():

        n_data = data_x.size(0)
        batch_indices = np.random.permutation(np.arange(n_data))
        batch_size = 512
        n_chunks = min(int(n_data/batch_size), 30)
        n_explore = 20 if n_explore is None else n_explore

        # where we store the function values
        eigindices = list(range(eigvals.shape[0])) if eigindices is None else eigindices
        f_values = t.zeros(len(eigindices), n_chunks, n_explore)

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

                direction = t.from_numpy(eigvecs[:, p]).to(data_x.device)

                for k in range(n_explore):

                    coef = 2.0 * (2*(k+1)/n_explore - 1)

                    z = model_params + coef * direction

                    z = model.shape_vec_as_params(z)
                    preds = functional_call(model, z, inputs)
                    loss = loss_fn(preds, targets)

                    f_values[index, i, k] = loss.item()

        return f_values


# line search around eigenvalues
if __name__ == "__main___":

    # data_x, data_y = t.load('models/resnet9_cifar10/enlarged_dataset.pth')

    # ======================================================================
    # Data Loading
    # ======================================================================

    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    data_train = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True)
    data_x = t.from_numpy(data_train.data).float()
    data_y = t.LongTensor(data_train.targets)

    x_mean = data_x.mean(dim=0)
    x_std = data_x.std(dim=0)

    data_x = (data_x - x_mean) / (1e-7 + x_std)
    data_x = data_x.transpose(1, 3)

    data_x = data_x.to(device)
    data_y = data_y.to(device)

    top_k = 1000
    iter = 10000

    eigvals, eigvecs = t.load(f"models/resnet9_cifar10/eig_{iter}/bottom_{top_k}_eigen.pth")

    # ======================================================================
    # Model loading
    # ======================================================================

    model = ResNet9(3, 10, expand_factor=1)
    model.load_state_dict(t.load(f'models/resnet9_cifar10/model_{iter}.pth'))

    gradient = t.load(f'./models/resnet9_cifar10/gradients/{iter}.pth')

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    f_values = line_search_around_eigenvalues(model, data_x, data_y, loss_fn, eigvals, eigvecs, gradient, n_explore=80)
    t.save(f_values, f'./models/resnet9_cifar10/line_searches/{iter}_neg_eigen.pth')


if __name__ == "__main__":
    # ======================================================================
    # plotting
    # ======================================================================

    import matplotlib.pyplot as plt
    import matplotlib

    cmap = matplotlib.colormaps['Spectral']

    iter = 10000

    # size [n_eigen, n_chunks, n_explore]
    f_values = t.load(f'./models/resnet9_cifar10/line_searches/{iter}_neg_eigen.pth')

    n_eigen = f_values.shape[0]
    n_chunks = f_values.shape[1]
    n_explore = f_values.shape[2]

    for i in range(n_eigen):

        total_f_values = np.sum(f_values[i].numpy(), axis=0)

        argmin_i = np.argmin(total_f_values)


        plt.figure(figsize=(14, 8))

        plt.plot(total_f_values, color='red', alpha=1.0, zorder=-1)

        # scatter the global minimum
        plt.scatter(argmin_i, total_f_values[argmin_i], color='blue', zorder=1)

        plt.savefig(f'./figures/neg_eigen_line_search_plots/total_{i}.png')
        plt.close()



    for i in range(n_eigen):

        argmin_i_values = [np.argmin(f_values[i, j].numpy()) for j in range(n_chunks)]
        min_values = [f_values[i,j, argmin_i_values[j]] for j in range(n_chunks)]

        sorted_indices = np.argsort(np.array(min_values))

        plt.figure(figsize=(14, 8))

        for j1 in sorted_indices:
            j = sorted_indices[j1]
            plt.plot(f_values[i, j].numpy(), color=cmap(j1/n_chunks), alpha=1.0, zorder=-1)
            min_i = np.argmin(f_values[i, j].numpy())

            # scatter the global minimum
            plt.scatter(min_i, f_values[i, j, min_i], color='blue', zorder=1)

            # scatter other local minimum

            # scatter local maximum

        plt.savefig(f'./figures/neg_eigen_line_search_plots/{i}.png')
        plt.close()


