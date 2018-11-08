from autograd import grad
import autograd.numpy as np


def logsumexp(x):
    """Numerically stable log(sum(exp(x)))"""
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

def example_func(y):
    z = y**2
    lse = logsumexp(z)
    return np.sum(lse)

grad_of_example = grad(example_func)
print ("Gradient: ", grad_of_example(np.array([1.5, 6.7, 1e-10])))
