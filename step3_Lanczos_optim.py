"""
The basic idea of Lanczos optim is to maintain an estimate of the k largest eigenvectors
of an optimisation problem, and then to use that basis to optimize the problem, and allow
much larger learning rates than would otherwise have been feasible.

basically, maintain k tentative eigenvectors, then choose 1, apply Hv to it, and do gram-schmidt from it
downwards, removing all components from higher matrices

there is a free parameter that determines the average number of lanczos updates per gradient
descent step.

todo: try a different approach: take n steps in the high eigenvalue space for every one step in the full space


"""
import torch as t
import torch.optim as optim
from HessianEigen_utils import *
import matplotlib.pyplot as plt

def orthogonalize(V, start_index=0):
    """
    :param V: t.Tensor of size [k, N]
    """

    n = V.size(0)
    norms = t.ones(n, device=V.device)

    norms[0] = V[0].norm()

    V[0] = V[0]/norms[0]

    for i in range(start_index+1, n):
        # orthonormal basis before V[i]
        prev_basis = V[0:i]

        # dots of each basis element with V[i], shape [i]
        coeff_vec = prev_basis @ V[i]

        # subtract projections of V[i] onto already determined basis V[0:i]
        V[i] -= coeff_vec @ prev_basis

        norms[i] = t.linalg.norm(V[i])

        V[i] /= norms[i]

    return V, norms


class Lanczos_optim:

    def __init__(self, model, top_k, lr=1e-3, weight_decay=1e-3, n_vec_updates_per_step=1, device=t.device('cuda')):

        self.top_k = top_k
        self.n_vec_updates_per_step = n_vec_updates_per_step
        self.hvp_eps = 1e-2
        self.model = model
        self.eigvecs, _ = orthogonalize(t.randn(top_k, self.model.n_params, device=device))
        self.eigvals = t.zeros(top_k)

        self.eig_gamma = 0.95
        self.eig_index = 0

        # initialize projections at current location of the model params
        self.proj_vars = (self.eigvecs @ self.params).detach()
        self.proj_vars.requires_grad = True
        self.params.requires_grad = True

        # self.optimizer = optim.AdamW([{'params': self.proj_vars, 'lr': lr/10},
        #                               {'params': self.params, 'lr':lr}], lr=lr, weight_decay=weight_decay)
        self.optimizer = optim.SGD(model.params(), lr=lr, momentum=0.97, weight_decay=weight_decay)
        self.high_eig_optimizer = optim.SGD(self.proj_vars)

    def fn_to_optim(self, p, inputs, targets, loss_fn):
        z = model.shape_vec_as_params(p)
        preds = functional_call(model, z, inputs)
        loss = loss_fn(preds, targets)
        return loss

    def update_eigvecs(self, eigvecs):

        self.eigvecs = eigvecs

    def lanczos_step(self, data_x, data_y, loss_fn):

        # ====================== Create Projected Params ========================

        # get param projections in a differentiable way
        projected_params = (self.params - (self.eigvecs @ self.params) @ self.eigvecs +
                            self.proj_vars @ self.eigvecs)
        # needed in order to access the gradient for future
        projected_params.retain_grad()

        # ====================== first backprop =================================

        # todo: do n optimization steps in the high-eigenvalue subspace with different batches

        # ====================== Normal Optim Step ==============================

        loss = loss_fn(self.model(inputs), targets)
        loss.backward()
        self.optimizer.step()


        # ======================= Update Model Params ===========================

        # now update the model with the new projected_parms
        for p1, p2 in zip(self.model.shape_vec_as_params_no_names(projected_params), self.model.parameters()):
            p2.data = p1.data.clone().detach()

        return loss


def get_validation(model, loss_fn, data_x, targets, top_k = 5):

    with t.no_grad():
        model.eval()

        top_k_accuracy = []

        preds = model(data_x)
        validation_loss = loss_fn(preds, targets)

        sorted_preds = t.argsort(preds, dim=1)

        correct_preds = sorted_preds[:, -1] == targets

        for i in range(2, top_k+2):

            accuracy = t.mean((correct_preds).float())
            top_k_accuracy.append(accuracy.item())

            correct_preds = t.logical_or(correct_preds, sorted_preds[:, -i] == targets)

        return validation_loss.item(), top_k_accuracy


if __name__ == "__main__":
    # define our device, send to gpu if we have it
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    # data_train = torchvision.datasets.MNIST('../datasets/', train=True, download=True)
    # data_test = torchvision.datasets.MNIST('../datasets/', train=False, download=True)
    data_train = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True)
    data_test = torchvision.datasets.CIFAR10('../datasets/', train=False, download=True)

    data_x = t.from_numpy(data_train.data).float()
    data_y = t.LongTensor(data_train.targets)

    x_mean = data_x.mean(dim=0)
    x_std = data_x.std(dim=0)

    data_x = (data_x - x_mean) / (1e-7 + x_std)

    test_data_x = data_test.data
    test_data_y = t.LongTensor(data_test.targets)

    test_data_x = t.from_numpy(test_data_x)
    test_data_x = test_data_x.float()
    test_data_x = (test_data_x - x_mean) / (1e-7 + x_std)

    lr = 1e-1
    weight_decay = 1e-3
    n_top = 50
    grad_clipping = 0.1
    n_top_updates = 50
    top_eigvals, top_eigvecs = None, None

    batch_size = 512
    n_epoch = 500
    n_data = data_x.size(0)
    n_iter = int(n_epoch * data_x.size(0) / batch_size)

    # ====================== Resnet MODEL =============================

    model = ResNet9(3, 10, expand_factor=1)
    model2 = ResNet9(3, 10, expand_factor=1)
    proj_vars = t.zeros(n_top).to(device)
    proj_vars.requires_grad = True
    print(f"number of parameters in model: {model.get_vectorized_params().shape}")
    loss_fn = nn.CrossEntropyLoss()

    # copy model into model2
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        p2.data = p1.data.clone().detach()

    # send things to the proper device
    model = model.to(device)
    model2 = model2.to(device)

    data_x = data_x.to(device).transpose(1, 3)
    data_y = data_y.to(device)

    test_data_x = test_data_x.to(device).transpose(1, 3)
    test_data_y = test_data_y.to(device)

    # defining optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    optimizer2 = optim.SGD(model2.parameters(), lr=lr, momentum=0.97, weight_decay=weight_decay)

    high_eigen_optimizer = optim.SGD([proj_vars], lr=1e-3, momentum=0)

    # plotting stuff
    plot_intervals = 100
    plot_average = None
    plot_average2 = None

    losses = []
    batch_indices_total = []

    plot_iters = []
    plot_losses = []
    plot_losses2 = []
    validation_loss = []
    last_accuracy = [0]

    start = time.time()
    for iter in range(n_iter):

        model.train()

        # ============================== Update eigenvectors ====================================

        if iter % 100 == 0:
            top_eigvals, top_eigvecs = top_k_hessian_eigen(model, data_x, data_y, loss_fn, top_k=n_top, mode='LA', batch_size=5000)

            print(f"top eigenvalue is {top_eigvals.max()}")

            for g in high_eigen_optimizer.param_groups:
                g['lr'] = 1e-3

        # ============================== Optimizie in high eigen subspace =========================

        params = None

        for i in range(n_top_updates):
            batch_indices = np.random.permutation(np.arange(n_data))[:batch_size]
            inputs = data_x[batch_indices]
            targets = data_y[batch_indices]


            def fn_to_optim(p):
                z = model.shape_vec_as_params(p)
                preds = functional_call(model, z, inputs)
                loss = loss_fn(preds, targets)
                return loss


            high_eigen_optimizer.zero_grad()
            params = model.get_vectorized_params().detach() + proj_vars @ t.from_numpy(top_eigvecs).to(device).T
            loss = fn_to_optim(params)
            loss.backward()
            nn.utils.clip_grad_value_([proj_vars], grad_clipping)
            high_eigen_optimizer.step()

        # clone finalized params
        for p1, p2 in zip(model.shape_vec_as_params_no_names(params), model.parameters()):
            p2.data = p1.data.clone().detach()

        # ============================= optimizing normal subspace =================================

        batch_indices = np.random.permutation(np.arange(n_data))[:batch_size]
        inputs = data_x[batch_indices]
        targets = data_y[batch_indices]

        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()

        optimizer2.zero_grad()
        loss2 = loss_fn(model2(inputs), targets)
        loss2.backward()
        optimizer2.step()

        # ============================================ Plotting Stuff ===============================================

        if iter % 1 == 0 and iter != 0:
            stop = time.time()

            time_remaining = (stop - start) * ((n_iter - iter) / iter)

            plot_average = loss
            plot_average2 = loss2

            print(
                f"i:{iter}/{n_iter} --- time remaining:{time_remaining:.1f}s --- loss:{plot_average.item():.7f} --- loss2:{loss2.item():.7f}"
                f" --- diff: {plot_average.item() - loss2.item(): .7f}")

        # changing learning rate once in a while
        if iter == 10000:
            for g in optimizer.param_groups:
                g['lr'] /= 10
            for g in optimizer2.param_groups:
                g['lr'] /= 10

        if iter == 20000:
            for g in optimizer.param_groups:
                g['lr'] /= 10
            for g in optimizer2.param_groups:
                g['lr'] /= 10

        if iter == 30000:
            for g in optimizer.param_groups:
                g['lr'] /= 10
            for g in optimizer2.param_groups:
                g['lr'] /= 10

        if iter == 40000:
            for g in optimizer.param_groups:
                g['lr'] /= 10
            for g in optimizer2.param_groups:
                g['lr'] /= 10


        if iter % plot_intervals == 0:
            plot_iters.append(iter)
            if plot_average is not None:
                plot_losses.append(plot_average.item())
                plot_losses2.append(plot_average2.item())

    # plot training loss, test loss and save figure
    plt.plot(np.array(plot_iters[:-1]), np.array(plot_losses), color='b', label='top eigen optim')
    plt.plot(np.array(plot_iters[:-1]), np.array(plot_losses2), color='r')
    plt.legend()
    plt.yscale('log')
    plt.show()
    plt.savefig('./figures/small_resnet_cifar10_top_eigen_optim.png')
