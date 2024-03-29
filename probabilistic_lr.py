
import torch as t
import torch.optim as optim
import numpy as np

def expected_value(x, y, eigen):

    exp_y = t.exp(y)
    exp_x = t.exp(x)

    y_midpoints = (exp_y[1:] + exp_y[:-1])/2
    x_midpoints = (exp_x[1:] + exp_x[:-1])/2
    Z = t.log(t.abs(1 - 2 * x_midpoints * eigen))

    x_deltas = (exp_x[1:] - exp_x[:-1])

    total_integral = t.sum(y_midpoints * x_deltas)

    expected_v = t.sum(x_midpoints * y_midpoints * x_deltas) / total_integral
    expected_log_term = t.sum(Z * y_midpoints * x_deltas) / total_integral

    return expected_v, expected_log_term


if __name__ == "__main__":

    # we parametrise the log space from -5 to 5 by 1000 sections

    n = 1000

    x = t.arange(-5, 5, 10/n)
    y = t.zeros(n)
    y.requires_grad = True


    # we minimize -E[alpha] + A * sum_eigen (E[log(1-2*alpha*eigen)] + epsilon)**2

    pass
