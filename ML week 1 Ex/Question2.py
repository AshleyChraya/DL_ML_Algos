# Question2.py
import numpy as np

import Question1 as q1  # importing First Question solution


def func1b(X, Y, epochs, threshold, LR):
    np.random.seed(1998)
    beta = np.random.rand(X.shape[1], 1)  # initial values of beta coefficients
    betainitial = beta
    Ypredict = np.matmul(X, beta)  # initial Y prediction
    init_cost = sum((Y - Ypredict) ** 2)  # initial cost function

    print("init cost", init_cost)
    for j in range(epochs):
        beta[0] = beta[0] + 2 * LR * sum(Y - np.matmul(X, beta))  # intercept
        for i in range(1, X.shape[1]):
            residualderivative = -2 * (Y - np.matmul(X, beta)) * np.array(X[:, i]).reshape(Y.shape[0], 1)
            beta[i] = beta[i] - LR * sum(residualderivative)

        Ypred = np.matmul(X, beta)  # Y prediction or the new betas
        new_cost = sum((Y - Ypred) ** 2)  # Cost function for the new betas
        print(new_cost)
        if init_cost - new_cost > threshold:  # condition
            init_cost = new_cost
        else:
            print(f"Loop break at {j}th iteration")
            break

    return betainitial, beta, new_cost, j


if __name__ == "__main__":
    std = 0.3
    n = 100
    m = 5
    X, Y, beta = q1.func1a(std, n, m)
    betainitial, beta, new_cost, iteration = func1b(X, Y, epochs=1000, threshold=0.000001, LR=0.001)
    print(f"vector of coefficients is m dimension: {beta.shape}")
    print(f"cost function value is {new_cost}")
