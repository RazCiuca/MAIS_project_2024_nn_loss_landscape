


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

def empirical_expected_loss(eigval, noise_std, lr, momentum):

    f_vals = []
    x = 0
    v = 0

    for i in range(1000):
        gradient = 2*eigval*(x + np.random.randn(1) * noise_std)

        v = momentum * v + gradient

        x -= lr * v

        f_vals.append(eigval * x**2)

    return np.mean(f_vals)



def line_search_around_eigenvalues(model, data_x, data_y, loss_fn, eigvals, eigvecs, eigindices=None, n_explore=None):
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
        n_chunks = min(int(n_data/batch_size), 15)
        n_explore = 20 if n_explore is None else n_explore

        # where we store the function values
        eigindices = list(range(eigvals.shape[0]))[::-1] if eigindices is None else eigindices
        f_values = t.zeros(len(eigindices), n_chunks, n_explore)

        explore_scale = 0.2
        desired_std = 10

        explore_scales = t.zeros(len(eigindices))
        std_minimums = t.zeros(len(eigindices))

        start = time.time()

        for index, p in enumerate(eigindices):
            stop = time.time()
            time_remaining = (stop - start) * ((len(eigindices) - index) / (1 if index == 0 else index))
            print(f"doing params {index}/{len(eigindices)}, time remaining:{time_remaining}")

            explore_scales[index] = explore_scale

            for i in range(n_chunks):

                indices = batch_indices[i*batch_size : (i+1)*batch_size]
                inputs = data_x[indices]
                targets = data_y[indices]

                direction = eigvecs[:, p].to(data_x.device)

                for k in range(n_explore):

                    coef = explore_scale * (2*(k+1)/n_explore - 1)

                    z = model_params + coef * direction
                    z = model.shape_vec_as_params(z)
                    preds = functional_call(model, z, inputs)
                    loss = loss_fn(preds, targets)

                    f_values[index, i, k] = loss.item()

            min_std = t.argmin(f_values[index], dim=1).float().std()
            std_minimums[index] = min_std
            print(f"variance in minimums: {min_std:.4f}")

            explore_scale *= min_std/desired_std

        return f_values, explore_scales, std_minimums, eigvals[eigindices]


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

    top_k = 10000
    iter = 10000

    eigvals, eigvecs = t.load(f"models/resnet9_cifar10/eig_{iter}/eigvals_vecs.pth")

    eigvals = eigvals[-top_k:]
    eigvecs = eigvecs[:, -top_k:]

    # ======================================================================
    # Model loading
    # ======================================================================

    model = ResNet9(3, 10, expand_factor=1)
    model.load_state_dict(t.load(f'models/resnet9_cifar10/model_{iter}.pth'))

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    f_values, explore_scales, std_min, eigvals_computed = line_search_around_eigenvalues(model, data_x, data_y, loss_fn, eigvals, eigvecs, n_explore=80)
    t.save((f_values, explore_scales, std_min, eigvals_computed), f'./models/resnet9_cifar10/line_searches/{iter}_pos_eigen.pth')


# plotting minimum variance against eigenvalue
if __name__ == "__main__":
    # ======================================================================
    # plotting minimum variance against eigenvalue
    # ======================================================================

    import matplotlib.pyplot as plt
    import matplotlib

    cmap = matplotlib.colormaps['Spectral']

    iter = 10000

    # size [n_eigen, n_chunks, n_explore]
    f_values, explore_scales, std_min, eigvals_computed = t.load(f'./models/resnet9_cifar10/line_searches/{iter}_pos_eigen.pth')

    n_eigen = f_values.shape[0]
    n_chunks = f_values.shape[1]
    n_explore = f_values.shape[2]

    true_std = std_min * explore_scales * 2 / n_explore

    plt.scatter(eigvals_computed.numpy(), true_std.numpy(), color='blue', alpha=0.1)
    plt.xlabel('eigenvalue')
    plt.ylabel('minimum std')
    plt.title('$\sigma$ of min location vs eigenvalue')
    plt.xscale('log')
    plt.show()

    # ======================================================================
    # plotting s visitation fraction against eigenvalue
    # ======================================================================

    lr = 0.1
    gamma = 0.97
    g = (1-gamma)*(1-gamma**2)

    eigvals = eigvals_computed.numpy()
    true_std = true_std.numpy()

    vis_var = lr * eigvals * true_std**2 / (1 - gamma**2 - lr*eigvals)

    plt.scatter(eigvals, vis_var**0.5, color='blue', alpha=0.1)
    plt.xlabel('eigenvalue')
    plt.ylabel('equilibrium visitation std')
    plt.title('$s$ of equilibirum distrbution vs eigenvalue')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    # ======================================================================
    # plotting total loss per eigenvalue
    # ======================================================================

    losses = vis_var*eigvals
    print('starting to compute losses')
    losses2 = np.array([empirical_expected_loss(eig, noise_std, 1e-1, 0.97) for eig, noise_std in zip(eigvals/2, true_std)])

    # plt.plot(eigvals, np.cumsum(losses[::-1])[::-1], label='expected cumul loss')
    plt.plot(eigvals, np.cumsum(losses2[::-1])[::-1], label='expected cumul loss')
    # plt.plot(eigvals, np.cumsum(losses2), label='expected cumul loss largest to smallest')


    plt.title(f'cumulative expected loss vs eigenvalue')
    plt.hlines(0.6, 1e-3, 1, colors='red', label='total loss of model')
    plt.legend()
    plt.xlabel('eigenvalue')
    plt.ylabel('expected loss')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

    # ======================================================================
    # plotting time to equilibrium
    # ======================================================================

    top_k = 10000
    iter = 10000

    eigvals, eigvecs = t.load(f"models/resnet9_cifar10/eig_{iter}/eigvals_vecs.pth")

    eigvals = eigvals[-top_k:]
    eigvecs = eigvecs[:, -top_k:]


    model = ResNet9(3, 10, expand_factor=1)
    model_init = ResNet9(3, 10, expand_factor=1)
    model.load_state_dict(t.load(f'models/resnet9_cifar10/model_{iter}.pth'))
    model_init.load_state_dict(t.load(f'models/resnet9_cifar10/model_0.pth'))

    total_param_change = model.get_vectorized_params() - model_init.get_vectorized_params()

    eigindices = list(range(eigvals.shape[0]))[::-1]
    projection = (total_param_change @ eigvecs).detach()
    projection = np.abs(projection[eigindices].numpy())

    tau = np.log(vis_var**0.5 / projection)/np.log(1.0 - 2*lr * eigvals_computed.numpy())

    plt.scatter(eigvals_computed.numpy(), tau, color='blue', alpha=0.1)
    plt.xlabel('eigenvalue')
    plt.ylabel('time to equilibrium')
    plt.title('time to equilibrium  vs eigenvalue')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()


