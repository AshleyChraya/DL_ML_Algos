# Question4.py
import matplotlib.pyplot as plt
import numpy as np

import Q1 as q1  # importing First Question solution
import Q2 as q2

if __name__ == "__main__":
    n = 100
    m = 5
    iter = []  # list to store at which epoch , we break the loop to get final new beta
    costf = []  # final cost functions list
    cos_similarity = []
    for theta in np.linspace(0, 1, 250):  # Variation with theta, 0<=theta<=1
        X, Y, beta_initial = q1.logisticregr(theta, n, m)
        beta, new_cost, iteration, cosinesimilarity = q2.gradientdescent(X, Y, beta_initial, 10000, 0.000001, 0.001)
        # print(cosinesimilarity)
        # iter.append(iteration)
        # costf.append(new_cost)
        cos_similarity.append(cosinesimilarity)
    print(cos_similarity)
    plt.plot(np.linspace(0, 1, 25), iter, marker="*", label="Iterations")
    plt.plot(np.linspace(0, 1, 25), costf, marker="*", label="Cost function")
    plt.plot(np.linspace(0, 1, 250), cos_similarity, marker="*", label="Cosine Similarity")
    plt.xlabel("theta")
    plt.title("n = 100 (fixed)")
    plt.legend()
    plt.show()