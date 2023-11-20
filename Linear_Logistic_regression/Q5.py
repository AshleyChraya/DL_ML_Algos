# Question5.py
import argparse  # Commandline input

import numpy as np
from numpy.linalg import norm

parser = argparse.ArgumentParser()
parser.add_argument(
    "-regression",
    type=str,
    default="linear",
    help="Put linear for linear regression and logisitic for logisitic regression",
)
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


class linear_logistic_reg:
    def __init__(self, std_dev, theta, n, m, epochs, threshold, LR):
        self.std_dev = std_dev
        self.theta = theta
        self.n = n
        self.m = m
        self.epochs = epochs
        self.threshold = threshold
        self.LR = LR

    def Regression(self):
        """Function to generate m*n independent variables, n*1 dependent variable, and coefficient matrix

        Args:
            stddev (float): spread of noise in the output variable (linear regression)
            theta (float): spread of noise in the output variable (logisitic regression)
            n (int): size of the data set
            m (int): number of indepedent variables

        Returns:
            array: independent, dependent variables matrix and coefficient matrix
        """
        x_ones = np.ones((self.n, 1))  # first column of ones
        # np.random.seed(1998)
        x_random = np.random.randn(self.n, self.m)  # (n*m) matrix of random Real numbers
        X = np.concatenate((x_ones, x_random), axis=1)  # n rows and m+1 columns where 1st column is of ones
        beta = np.random.rand(self.m + 1, 1)  # Coefficients

        if args.regression == "linear":
            error = np.random.normal(loc=0, scale=self.std_dev, size=(self.n, 1))
            Y = (
                np.matmul(X, beta) + error
            )  # X*beta i.e. matrix multiplication of (n*m) and (m * 1) shape= Y shape (n*1)

        if args.regression == "logistic":
            Y = (
                1 / (1 + np.exp(-np.dot(X, beta))) >= 0.5
            )  # X*beta i.e. matrix multiplication of (n*m) and (m * 1) shape= Y shape (n*1)

            Y = Y.astype(int)
            noise = np.random.binomial(n=1, p=self.theta, size=(self.n, 1))
            Y = Y + noise
            Y %= 2

        return X, Y, beta


    def cost(self, Y_pred, Y_true, beta):
        """Calculate the cost

        Args:
            Y_pred (array): predicted value of Y output
            beta (array): value of beta array at current iteration including initial initialization

        Returns:
            float: cost at current iteration
        """

        # remove 0 and 1 from Y_pred and convert it to close to 0 and 1 respectively
        Y_pred = np.where(Y_pred == 0, 1e-4, Y_pred)
        Y_pred = np.where(Y_pred == 1, 1 - 1e-4, Y_pred)

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

    def gradientdescent(self, X, Y_true):
        np.random.seed(1998)
        beta = np.random.rand(X.shape[1], 1)  # initial values of beta coefficients
        print("Initial beta ", beta)

        if args.regression == "linear":
            Ypredict = np.matmul(X, beta)  # initial Y prediction
            init_cost = sum((Y_true - Ypredict) ** 2)  # initial cost function

            print("init cost", init_cost)
            for j in range(self.epochs):
                beta[0] = beta[0] + 2 * self.LR * sum(Y_true - np.matmul(X, beta))  # intercept
                for i in range(1, X.shape[1]):
                    residualderivative = (
                        -2 * (Y_true - np.matmul(X, beta)) * np.array(X[:, i]).reshape(Y_true.shape[0], 1)
                    )
                    beta[i] = beta[i] - self.LR * sum(residualderivative)

                Ypred = np.matmul(X, beta)  # Y prediction or the new betas
                new_cost = sum((Y_true - Ypred) ** 2)  # Cost function for the new betas
                print(new_cost)
                if init_cost - new_cost > self.threshold:  # condition
                    init_cost = new_cost
                else:
                    print(f"Loop break at {j}th iteration")
                    break
            cosinesim_Y = np.dot(Y_true.flatten(), Ypred.flatten()) / (norm(Y_true.flatten()) * norm(Ypred.flatten()))

            return beta, new_cost, j, cosinesim_Y

        if args.regression == "logistic":

            Y_pred = 1 / (1 + np.exp(-np.dot(X, beta)))  # initial Y prediction

            init_cost = test.cost(Y_pred, Y_true, beta)  # initial cost function

            # print("init cost", init_cost)
            for j in range(self.epochs):
                if args.regularization == "0":
                    beta[0] = beta[0] + self.LR * sum(Y_true - Y_pred)  # intercept

                    for i in range(1, X.shape[1]):
                        residualderivative = -(Y_true - Y_pred) * np.array(X[:, i]).reshape(Y_true.shape[0], 1)
                        beta[i] = beta[i] - self.LR * sum(residualderivative)

                if args.regularization == "l1":
                    beta[0] = (
                        beta[0] + self.LR * sum(Y_true - Y_pred) - args.regconstant * beta[0] / np.abs(beta[0])
                    )  # intercept

                    for i in range(1, X.shape[1]):
                        residualderivative = -(Y_true - Y_pred) * np.array(X[:, i]).reshape(
                            Y_true.shape[0], 1
                        ) + args.regconstant * beta[i] / np.abs(beta[i])
                        beta[i] = beta[i] - self.LR * sum(residualderivative)

                if args.regularization == "l2":
                    beta[0] = beta[0] + self.LR * sum(Y_true - Y_pred) - args.regconstant * 2 * beta[0]  # intercept

                    for i in range(1, X.shape[1]):
                        residualderivative = (
                            -2 * (Y_true - Y_pred) * np.array(X[:, i]).reshape(Y_true.shape[0], 1)
                            + args.regconstant * 2 * beta[i]
                        )
                        beta[i] = beta[i] - self.LR * sum(residualderivative)

                Y_pred = 1 / (1 + np.exp(-np.dot(X, beta)))  # Y prediction for the new betas

                new_cost = test.cost(Y_pred, Y_true, beta)  # Cost function for the new betas
                # print(new_cost)
                if init_cost - new_cost > self.threshold:  # condition
                    init_cost = new_cost
                    if j == self.epochs - 1:
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

    test = linear_logistic_reg(std_dev=0.3, theta=0.8, n=1000, m=50, epochs=10000, threshold=0.00001, LR=0.001)
    X, Y_true, beta_true = test.Regression()
    print(f"True (Original) beta vector: {beta_true}")
    print(f"True (Original) output vector: {Y_true}")
    beta_new, new_cost, iteration, cosinesim_Y = test.gradientdescent(
        X, Y_true
    )
    print(f"Obtained beta vector after gradient descent: {beta_new}")
    print(f"cost function value is {new_cost}")
    print(f"Cosine similarity between True Y vector and Y vector after gradient descent is {cosinesim_Y}")
