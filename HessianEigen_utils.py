"""
Razvan Ciuca, 2024

Contains useful functions for computing Hessians of models implicitely, as well
as optimising in models in defined subspaces and computing the top/bottom eigenvalues and vectors
of models

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

def hvp_func(f, primals, tangents):
  return jvp(grad(f), primals, tangents)[1:]

# todo: finish this function, this doesn't work now
def hvp_model(tangents, model, data_x, data_y, loss_fn, chunk_size=8):

    # send data to gpu
    data_x = data_x.cuda()
    data_y = data_y.cuda()

    # defining the function to send to hvp
    def fn_to_optim(*x):
        z = [{name: p} for (p, name) in zip(x, model.param_names)]

        preds = functional_call(model, z, data_x)
        return loss_fn(preds, data_y)

    def to_vectorize(x):
        z = tuple(model.shape_vec_as_params_no_names(x))
        vh_prod = hvp_func(fn_to_optim, tuple(model.parameters()), z)
        return t.cat([z.flatten() for z in vh_prod])

    total_hvp = t.vmap(to_vectorize, in_dims=0, chunk_size=chunk_size)(tangents)

    return total_hvp


def hvp_model_np(tangents_np, model, data_x, data_y, loss_fn, chunk_size=None):
    tangents = t.from_numpy(tangents_np).to('cuda').squeeze()
    tangents = tuple(model.shape_vec_as_params_no_names(tangents))

    # send data to gpu
    data_x_tot = data_x.cuda()
    data_y_tot = data_y.cuda()
    n_data = data_x_tot.size(0)

    res = np.zeros(tangents_np.shape[0])

    n_iter = 1 if chunk_size is None else (int(n_data/chunk_size)+1)
    for i in range(n_iter):

        data_x = data_x_tot[i*chunk_size: (i+1)*chunk_size]
        data_y = data_y_tot[i*chunk_size: (i+1)*chunk_size]

        # defining the function to send to hvp
        def fn_to_optim(*x):
            z = [{name: p} for (p, name) in zip(x, model.param_names)]

            preds = functional_call(model, z, data_x)
            return loss_fn(preds, data_y)

        hessian_col = vhp(fn_to_optim, tuple(model.parameters()), tangents)[1]

        res += t.cat([x.flatten() for x in hessian_col]).detach().cpu().numpy() * data_x.size(0)

    return res/n_data

def grad_model(model, data_x, data_y, loss_fn):

    # send data to gpu
    data_x = data_x.cuda()
    data_y = data_y.cuda()

    preds = model(data_x)

    loss = loss_fn(preds, data_y)

    gradients = t.autograd.grad(loss, model.parameters())

    return t.cat([x.flatten() for x in gradients]).detach().cpu()

def top_k_hessian_eigen(model, data_x, data_y, loss_fn, top_k = 100, mode='LA', batch_size=None, v0=None, chunk_size=None):
    """
    computes top-k eigenvalues and eigenvectors of the hessian of model, with given data,
    possibly with finite batch size
    """
    # if finite batch size, resample this at every computation?
    if batch_size is not None:
        indices = t.randperm(data_x.size(0), device=data_x.device)[:batch_size]
        data_x = data_x[indices]
        data_y = data_y[indices]


    linop = LinearOperator((model.n_params, model.n_params),
                            matvec = lambda x: hvp_model_np(x, model, data_x, data_y, loss_fn, chunk_size))

    eigvals, eigvecs = eigsh(linop, k=top_k, which=mode, v0=v0)

    n_samples = data_x.size(0)

    return eigvals, eigvecs

def update_eigenvecs_estimate(eigvecs_estimate, indices, hv_updates):
    """

    :param eigvecs_estimate: [n_params, n_eig] vectors estimates of norm 1
    :param indices: indices at which to update
    :param hv_updates: the hessian vector product with respect to the eigvecs at indices
    :return:
    """

    # remove the components from 0 to i-1 from i

    # remove components along i from i+ 1 to n

    pass


def make_line_search_dir(eigvals, eigvecs):
    # take all the negative eigvals, and scale the corresponding vectors by that, in order
    # to give a direction where the decrease in loss is uniformly spread across negative eigenvalues
    mask = eigvals < 0
    mask_val = -eigvals[mask]
    mask_vec = (eigvecs.T)[mask]

    # seach direction indexed by t, each direction should be sqrt(t/eig)

    def get_direction_fn(c):
        c = t.Tensor([c]).to(eigvecs.device)
        return (c/mask_val) @ mask_vec

    return get_direction_fn

def line_search(model, loss_fn, inputs, targets, eigvals, eigvecs):
    """
    does line search in a direction defined by the eigenvalues
    """
    flat_params = model.get_vectorized_params()

    get_direction_fn = make_line_search_dir(eigvals, eigvecs)

    log_c_ar = np.arange(-46, -45, 0.001)
    loss_ar = []

    minimum = np.inf
    min_loc = None

    for log_c in log_c_ar:
        with t.no_grad():

            x = flat_params + get_direction_fn(np.exp(log_c))
            z = model.shape_vec_as_params(x)

            # z = [{name: p} for (p, name) in zip(x, model.param_names)]

            preds = functional_call(model, z, inputs)
            loss = loss_fn(preds, targets)
            loss_ar.append(loss.item())

            print(f"log-c: {log_c:.2f}, loss:{loss.item()}")

            if loss.item() < minimum:
                minimum = loss.item()
                min_loc = log_c

    # spl = CubicSpline(loss_ar, np.array(loss_ar))
    #
    # optimal_c = minimize(spl, np.array([min_loc])).x

    optimal_x = flat_params + get_direction_fn(np.exp(min_loc))

    return model.shape_vec_as_params(optimal_x)

def sgd_in_subspace(model, loss_fn, inputs, targets, eigvals, eigvecs, device, n_iter=1000):
    """
    does SGD in the space defined by eigvecs, as a difference of parameters from the current model
    """
    # parametrise the space by the negative eigenvectors, then do gradient descent in that
    # mask = eigvals < 0
    # mask_val = -eigvals[mask]
    # mask_vec = (eigvecs.T)[mask]
    mask_val = t.from_numpy(eigvals).to(device)
    mask_vec = t.from_numpy(eigvecs).T.to(device)

    optim_x = t.zeros(mask_val.size(0)).to(device)
    optim_x.requires_grad = True

    flat_params = model.get_vectorized_params().detach()

    optimizer = optim.Adam([optim_x], lr=1e-3, betas=(0.9, 0.99))
    # optimizer = optim.SGD([optim_x], lr=1e-2, momentum=0.9)
    # optimizer = t.optim.LBFGS([optim_x], lr=1.0, history_size=1000, line_search_fn='strong_wolfe')
    last_loss = 0

    for i in range(n_iter):

        x = flat_params + optim_x @ mask_vec
        z = model.shape_vec_as_params(x)
        # z = [{name: p} for (p, name) in zip(x, model.param_names)]

        preds = functional_call(model, z, inputs)
        loss = loss_fn(preds, targets)

        if i%10 == 0:
            print(f"i:{i}/{n_iter} --- loss:{loss.item():.9f} --- loss delta: {last_loss - loss.item():.3e}")
            last_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model.shape_vec_as_params_no_names(flat_params + optim_x @ mask_vec)

def evaluate_model(params, model, inputs, targets, loss_fn):

    with t.no_grad():
        ave_loss = 0

        z = model.shape_vec_as_params(params)
        preds = functional_call(model, z, inputs)
        loss = loss_fn(preds, targets)
        ave_loss += loss.item()

        return ave_loss

def model_contour_plot(model, get_params, inputs, targets, loss_fn, x_vals, y_vals):

    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros([len(x_vals), len(y_vals)])

    init_params = model.get_vectorized_params()

    for i1, x in enumerate(x_vals):
        for i2, y in enumerate(y_vals):

            # params = init_params + x * vec_x + y * vec_y
            params = get_params(x, y)

            loss = evaluate_model(params, model, inputs, targets, loss_fn)

            Z[i1,i2] = loss

            print(f"done ({i1}/{len(x_vals)},{i2}/{len(y_vals)}), loss:{loss:.6f}")

    return X, Y, Z



class HessianCompute:

    def __init__(self, dataset, model, loss_fn):
        super(HessianCompute, self).__init__()

        self.data_x, self.data_y = dataset
        self.n_data = self.data_x.size(0)
        self.loss_fn = loss_fn
        self.model = model
        self.params = tuple(model.parameters())
        self.vec_params = self.model.get_vectorized_params()
        self.n_params = self.vec_params.size(0)

        self.batch_chunksize = 15000
        self.n_chunks = int(self.data_x.size(0)/self.batch_chunksize)+1

        self.chunk_size_to_save = 10**9
        self.save_every = 1000

        self.chunked_data = []

        for k in range(self.n_chunks):
            j0, j1 = k * self.batch_chunksize, (k + 1) * self.batch_chunksize

            # send data to gpu
            data_x = self.data_x[j0:j1].cuda()
            data_y = self.data_y[j0:j1].cuda()

            self.chunked_data.append((data_x, data_y))


    def compute_hessian(self, folder_to_save):
        """
        iterate over the dataset, computing hessian vector products, then sending them to gpu
        :return:
        """
        chunk_counter = 0

        total_hess = []

        # for loop over columns of the Hessian
        start = time.time()

        model.train()

        for i in range(self.n_params):
            # for loop over chunks of the dataset

            tangents = t.zeros(self.n_params, device='cuda')
            tangents[i] = 1
            tangents = tuple(self.model.shape_vec_as_params_no_names(tangents))

            total_col = t.zeros(self.n_params, device='cuda')

            for k in range(self.n_chunks):
                data_x, data_y = self.chunked_data[k]

                # defining the function to send to hvp
                def fn_to_optim(*x):

                    z = [{name: p} for (p, name) in zip(x, self.model.param_names)]

                    preds = functional_call(self.model, z, data_x)
                    return self.loss_fn(preds, data_y)

                # calling hvp with the right vector, add to result
                hessian_col = vhp(fn_to_optim, self.params, tangents)[1]
                total_col += data_x.size(0)*t.cat([x.flatten() for x in hessian_col])

            # bring it back to cpu
            total_hess.append(total_col.cpu()/self.n_data)
            if i%10 == 0 and i != 0:
                stop = time.time()

                time_remaining = (stop - start) * ((self.n_params - i) / i)
                print(f"finished {i}/{self.n_params} --- time remaining:{time_remaining:.1f}s ")

            if i%self.save_every == 0 and i != 0:
                t.save(t.stack(total_hess, dim=1), folder_to_save + f'/hess_{chunk_counter}')
                chunk_counter += 1
                total_hess = []

        t.save(t.stack(total_hess, dim=1), folder_to_save + f'/hess_{chunk_counter}')

def find_direction_of_greatest_eigval_increase():

    pass



def load_hessian_from_folder(foldername):

    pass

# bottom eigen compute
if __name__ == "__main__":

    data_train = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True)
    data_x = t.from_numpy(data_train.data).float()
    data_y = t.LongTensor(data_train.targets)

    x_mean = data_x.mean(dim=0)
    x_std = data_x.std(dim=0)

    data_x = (data_x - x_mean) / (1e-7 + x_std)
    data_x = data_x.transpose(1, 3)

    model = ResNet9(3, 10, expand_factor=1)
    model.load_state_dict(t.load(f'models/resnet9_cifar10/model_{10000}.pth'))

    model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()

    eigvals, eigvecs = top_k_hessian_eigen(model, data_x, data_y, loss_fn, top_k = 10, mode='SA', batch_size=10000)

    print(eigvals)


# total hessian compute
if __name__ == "__main___":

    # data_x, data_y = t.load('models/resnet9_cifar10/enlarged_dataset.pth')

    data_train = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True)
    data_x = t.from_numpy(data_train.data).float()
    data_y = t.LongTensor(data_train.targets)

    x_mean = data_x.mean(dim=0)
    x_std = data_x.std(dim=0)

    data_x = (data_x - x_mean) / (1e-7 + x_std)
    data_x = data_x.transpose(1, 3)

    for iters in [100, 200, 500, 10100, 10200, 10500]:

        model = ResNet9(3, 10, expand_factor=1)
        model.load_state_dict(t.load(f'models/resnet9_cifar10/model_{iters}.pth'))

        model = model.cuda()
        loss_fn = nn.CrossEntropyLoss()

        dataset = (data_x, data_y)

        hess_comp = HessianCompute(dataset, model, loss_fn)
        hess_comp.compute_hessian(folder_to_save=f'./models/resnet9_cifar10/hess_{iters}')



# compute model gradients
if __name__ == "__main___":

    # data_x, data_y = t.load('models/resnet9_cifar10/enlarged_dataset.pth')

    data_train = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True)
    data_x = t.from_numpy(data_train.data).float()
    data_y = t.LongTensor(data_train.targets)

    x_mean = data_x.mean(dim=0)
    x_std = data_x.std(dim=0)

    data_x = (data_x - x_mean) / (1e-7 + x_std)

    for iters in [48800, 20000, 2000, 0]:

        model = ResNet9(3, 10, expand_factor=1)
        model.load_state_dict(t.load(f'models/resnet9_cifar10/model_{iters}.pth'))

        model = model.cuda()
        loss_fn = nn.CrossEntropyLoss()

        data_x = data_x.transpose(1,3)
        data_y = data_y

        gradient = grad_model(model, data_x, data_y, loss_fn)
        t.save(gradient, f'./models/resnet9_cifar10/gradients/{iters}.pth')
