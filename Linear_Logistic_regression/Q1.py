# Question1.py
# import random

import numpy as np


def logisticregr(theta, n, m):
    """Function to generate n*(m+1) independent variables, n*1 dependent variable, and coefficient matrix (m+1*1)

    Args:
        theta (float): spread of noise in the output variable
        n (int): size of the data set
        m (int): number of indepedent variables

    Returns:
        array: independent, dependent variables matrix and coefficient matrix
    """
    x_ones = np.ones((n, 1))  # first column of ones
    # np.random.seed(1998)
    x_random = np.random.randn(n, m)  # (n*m) matrix of random Real numbers
    X = np.concatenate((x_ones, x_random), axis=1)  # n rows and m+1 columns where 1st column is of ones
    beta = np.random.rand(m + 1, 1)  # Coefficients
    Y = (
        1 / (1 + np.exp(-np.dot(X, beta))) >= 0.5
    )  # X*beta i.e. matrix multiplication of (n*m) and (m * 1) shape= Y shape (n*1)

    Y = Y.astype(int)
    noise = np.random.binomial(n=1, p=theta, size=(n, 1))
    Y = Y + noise
    Y %= 2

    ##### ------------------------------- ####
    # Alternatively, we can use Maximum likelihood estimation to add noise to Y

    # probflip = theta  # probability that the output would be flipped
    # # As it is a bernoulli problem, by Maximum likelihood estimation, we know (theta = # of flips/ n)
    # # hence, total number of flips =  n*theta)...taking this number of random sets of inputs whose output is reversed

    # # flipping outputs of Y (adding noise)
    # for i in random.sample(range(0, n), int(n * probflip)):  # n unique random intergers
    #     if Y[i] == 1:
    #         Y[i] = 0
    #     else:
    #         Y[i] = 1

    #### ---------------------------------- ######

    return X, Y, beta


if __name__ == "__main__":
    theta = 0.3
    n = 10
    m = 5
    X, Y_true, beta_true = logisticregr(theta, n, m)
    print("shape of X is ", X.shape)
    print("shape of beta is ", beta_true.shape)
    print("shape of Y is ", Y_true.shape)
    print("X matrix is \n", X)
    print("beta vector is \n", beta_true)
    print("Y dependent vector is \n ", Y_true)
