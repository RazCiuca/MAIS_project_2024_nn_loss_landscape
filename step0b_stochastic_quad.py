"""

does SGD on a pure quadratic function with added noise, to see what the loss landscape looks like

"""

import numpy as np
import matplotlib.pyplot as plt

# our optimisation variable
x = np.array([1.0])

x_no_momentum = np.array([1.0])

# we're optimising f(x) = alpha * x^2
alpha = 1

# learning rate
lr = 0.01
gamma = 0.8

# this is the gradient noise
noise_std = 0.05

f_values = []
f_values_no_momentum = []

# plot exponential moving average
gamma_plot = 0.9
f_moving_average = 0
f_moving_average_no_momentum = 0
v = 0

for i in range(2000):

    # append values once in a while
    fn_val = alpha * x ** 2
    f_moving_average = gamma_plot * f_moving_average + (1.0 - gamma_plot) * fn_val
    f_values.append(f_moving_average / (1.0 - gamma_plot ** (i + 1)))

    f_moving_average_no_momentum = gamma_plot * f_moving_average_no_momentum + (1.0 - gamma_plot) * alpha * (x_no_momentum)**2

    f_values_no_momentum.append(f_moving_average_no_momentum/(1.0 - gamma_plot ** (i + 1)))

    # compute gradient and add noise
    random_factor = np.random.randn(1)
    gradient = 2 * alpha * (x - noise_std * random_factor)

    x_no_momentum = x_no_momentum - lr * 2 * alpha * (x_no_momentum - noise_std * random_factor)

    # do gradient descent with noise
    v = gamma * v + gradient
    x -= lr * v


expected_loss = lr * alpha**2 * noise_std**2 / (1 - gamma**2 - lr*alpha)

plt.plot(np.array(f_values), color='blue', label='sgd with momentum')
plt.plot(np.array(f_values_no_momentum), color='purple', label='sgd')

plt.hlines(expected_loss, xmin=0, xmax=len(f_values), color='red', label='expected loss sgd with momentum')
plt.hlines(lr * alpha**2 * noise_std**2 / (1 - lr*alpha), xmin=0, xmax=len(f_values), color='orange', label='expected loss sgd')

plt.xlabel('iteration')
plt.ylabel('loss')
plt.yscale('log')
plt.legend()
plt.title('SGD with momentum vs no momentum')
plt.show()