# Question2 and Question 4.py
import argparse  # Commandline input

import numpy as np
from numpy.linalg import norm

import Q1 as q1  # importing First Question solution

parser = argparse.ArgumentParser()
parser.add_argument(
    "-regularization",
    type=str,
    default="0",
    help="Put l1 for lasso or l2 for Ridge or 0 for no regularization",
)
parser.add_argument(
    "-regconstant",
    type=float,
    help="Regularization constant (float)",
)
args = parser.parse_args()


def cost(Y_pred, Y_true, beta):
    """Calculate the cost

    Args:
        Y_pred (array): predicted value of Y output
        beta (array): value of beta array at current iteration including initial initialization

    Returns:
        float: cost at current iteration
    """

    # remove 0 and 1 from Y_pred and convert it to close to 0 and 1 respectively
    Y_pred = np.where(Y_pred == 0, 0.01, Y_pred)
    Y_pred = np.where(Y_pred == 1, 0.99, Y_pred)

    # removing any 0 component in beta such that 0 doesn't come in denomenator in regularization term
    beta = np.where(beta == 0, 0.01, beta)

    if args.regularization == "0":

        cost = -np.mean(Y_true * np.log(Y_pred) + (1 - Y_true) * (np.log(1 - Y_pred)))  # initial cost function

    elif args.regularization == "l1":
        cost = -np.mean(Y_true * np.log(Y_pred) + (1 - Y_true) * (np.log(1 - Y_pred))) + args.regconstant * np.sum(
            np.abs(beta)
        )  # initial cost function
    elif args.regularization == "l2":
        cost = -np.mean(Y_true * np.log(Y_pred) + (1 - Y_true) * (np.log(1 - Y_pred))) + args.regconstant * np.sum(
            beta**2
        )

    return cost


def gradientdescent(X, Y_true, epochs, threshold, LR):
    np.random.seed(1998)
    beta = np.random.rand(X.shape[1], 1)  # initial values of beta coefficients
    # print("Initial beta ", beta)

    Y_pred = 1 / (1 + np.exp(-np.dot(X, beta)))  # initial Y prediction

    init_cost = cost(Y_pred, Y_true, beta)  # initial cost function

    # print("init cost", init_cost)
    for j in range(epochs):
        if args.regularization == "0":
            beta[0] = beta[0] + LR * sum(Y_true - Y_pred)  # intercept

            for i in range(1, X.shape[1]):
                residualderivative = -(Y_true - Y_pred) * np.array(X[:, i]).reshape(Y_true.shape[0], 1)
                beta[i] = beta[i] - LR * sum(residualderivative)

        if args.regularization == "l1":
            beta[0] = beta[0] + LR * sum(Y_true - Y_pred) - args.regconstant * beta[0] / np.abs(beta[0])  # intercept

            for i in range(1, X.shape[1]):
                residualderivative = -(Y_true - Y_pred) * np.array(X[:, i]).reshape(
                    Y_true.shape[0], 1
                ) + args.regconstant * beta[i] / np.abs(beta[i])
                beta[i] = beta[i] - LR * sum(residualderivative)

        if args.regularization == "l2":
            beta[0] = beta[0] + LR * sum(Y_true - Y_pred) - args.regconstant * 2 * beta[0]  # intercept

            for i in range(1, X.shape[1]):
                residualderivative = (
                    -2 * (Y_true - Y_pred) * np.array(X[:, i]).reshape(Y_true.shape[0], 1)
                    + args.regconstant * 2 * beta[i]
                )
                beta[i] = beta[i] - LR * sum(residualderivative)

        Y_pred = 1 / (1 + np.exp(-np.dot(X, beta)))  # Y prediction for the new betas

        new_cost = cost(Y_pred, Y_true, beta)  # Cost function for the new betas
        # print(new_cost)
        if init_cost - new_cost > threshold:  # condition
            init_cost = new_cost
            if j == epochs - 1:
                print(f"Loop ran till last epoch {j}")
        else:
            print(f"Loop break at {j}th iteration")
            break
    # cosinesim_beta = np.dot(beta_true.flatten(), beta.flatten()) / (norm(beta_true.flatten()) * norm(beta.flatten()))
    Y_pred = Y_pred >= 0.5
    Y_pred = Y_pred.astype(int)

    cosinesim_Y = np.dot(Y_true.flatten(), Y_pred.flatten()) / (norm(Y_true.flatten()) * norm(Y_pred.flatten()))
    return beta, new_cost, j, cosinesim_Y


if __name__ == "__main__":

    theta = 0.8
    n = 1000
    m = 50
    X, Y_true, beta_true = q1.logisticregr(theta, n, m)
    print(f"True (Original) beta vector: {beta_true}")
    print(f"True (Original) output vector: {Y_true}")
    beta_new, new_cost, iteration, cosinesim_Y = gradientdescent(X, Y_true, epochs=10000, threshold=0.00001, LR=0.001)
    print(f"Obtained beta vector after gradient descent: {beta_new}")
    print(f"cost function value is {new_cost}")
    # print(f"Cosine similarity between True beta vector and beta vector after gradient descent is {cosinesim_beta}")
    print(f"Cosine similarity between True Y vector and Y vector after gradient descent is {cosinesim_Y}")
