# Question1.py
import numpy as np


def func1a(stddev, n, m):
    """Function to generate m*n independent variables, n*1 dependent variable, and coefficient matrix

    Args:
        stddev (float): spread of noise in the output variable
        n (int): size of the data set
        m (int): number of indepedent variables

    Returns:
        array: independent, dependent variables matrix and coefficient matrix
    """
    x_ones = np.ones((n, 1))  # first column of ones
    np.random.seed(1998)
    x_random = np.random.rand(n, m - 1)  # (n*m-1) matrix of random Real numbers
    X = np.concatenate((x_ones, x_random), axis=1)  # n rows and m columns where 1st column is of ones
    beta = np.random.rand(m, 1)  # Coefficients
    error = np.random.normal(loc=0, scale=stddev, size=(n, 1))
    Y = np.matmul(X, beta) + error  # X*beta i.e. matrix multiplication of (n*m) and (m * 1) shape= Y shape (n*1)
    return X, Y, beta


if __name__ == "__main__":
    std = 0.3
    n = 100
    m = 5
    X, Y, beta = func1a(std, n, m)
    print("shape of X is ", X.shape)
    print("shape of beta is ", beta.shape)
    print("X matrix is \n", X)
    print("beta vector is \n", beta)
    print("Y dependent vector is \n ", Y)
    print("shape of Y is ", Y.shape)
