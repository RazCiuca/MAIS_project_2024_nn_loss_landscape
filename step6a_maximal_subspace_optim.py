"""
The goal of this file is to test the minimum loss achievable
when doing sgd only in the components with the largest magnitude.

So here's what we do:
- warmup the optimisation for a few hundred iterations, from a 0 learning rate to the starting one.
- then compute the top n eigenvalues and the bottom n eigenvalues
- test the following optimisation schemes:
    - only top
    - only bottom
    - both top and bottom
- and see what the smallest loss achievable is on full batch learning

"""


