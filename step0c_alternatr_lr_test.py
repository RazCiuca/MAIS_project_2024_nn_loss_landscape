"""

does SGD on a pure quadratic function with added noise, to see what the loss landscape looks like

"""

import numpy as np
import matplotlib.pyplot as plt

# our optimisation variable
x = np.array([1.0])

# we're optimising f(x) = alpha * x^2
alpha = 1

# learning rate
lr1 = 0.40
lr2 = 20
gamma = 0.0

# this is the gradient noise
noise_std = 0.05

f_values = []

# plot exponential moving average
gamma_plot = 0.9
f_moving_average = 0
v = 0

for i in range(2000):

    lr = lr1 if (i % 3) != 0 else lr2

    # append values once in a while
    fn_val = alpha * x ** 2
    f_moving_average = gamma_plot * f_moving_average + (1.0 - gamma_plot) * fn_val
    f_values.append(f_moving_average / (1.0 - gamma_plot ** (i + 1)))

    # compute gradient and add noise
    random_factor = np.random.randn(1)
    gradient = 2 * alpha * (x - noise_std * random_factor)

    # do gradient descent with noise
    v = gamma * v + gradient
    x -= lr * v


plt.plot(np.array(f_values), color='blue', label='sgd')

plt.xlabel('iteration')
plt.ylabel('loss')
plt.yscale('log')
plt.legend()
plt.title('SGD with alternate lr')
plt.show()