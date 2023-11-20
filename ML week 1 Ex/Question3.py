# Question3.py
import matplotlib.pyplot as plt
import numpy as np

import Question1 as q1  # importing First Question solution
import Question2 as q2

if __name__ == "__main__":
    n = 100
    m = 5
    iter = []
    costf = []
    for std in np.linspace(0.0001, 5, 25):  # Variation with std dev
        X, Y, beta = q1.func1a(std, n, m)
        beta, new_cost, iteration = q2.func1b(X, Y, 1000, 0.000001, 0.001)
        iter.append(iteration)
        costf.append(new_cost)

    plt.plot(np.linspace(0.01, 5, 25), iter, marker="*", label="Iterations")
    plt.plot(np.linspace(0.01, 5, 25), costf, marker="*", label="Cost function")
    plt.xlabel("standard deviation")
    plt.title("n = 100 (fixed)")
    plt.legend()
    plt.show()

    std = 1
    m = 5
    iter = []
    costf = []
    for n in range(10, 2000, 100):  # Variation with n
        X, Y, beta = q1.func1a(std, n, m)
        beta, new_cost, iteration = q2.func1b(X, Y, 1000, 0.000001, 0.001)
        iter.append(iteration)
        costf.append(new_cost)

    plt.plot(range(10, 2000, 100), iter, marker="*", label="Iterations")
    plt.plot(range(10, 2000, 100), costf, marker="*", label="Cost function")
    plt.xlabel("n")
    plt.title("std dev = 1 (fixed)")
    plt.legend()
    plt.show()
