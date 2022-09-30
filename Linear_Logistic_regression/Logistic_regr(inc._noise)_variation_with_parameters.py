# Question3.py
import matplotlib.pyplot as plt
import numpy as np

import Q1 as q1  # importing First Question solution
import Q2_Q4 as q2

# ---------Caveat----------------    #

# 1) I have divided the script into three parts, please run one by one and comment others which is not needed for the moment
# 2) section 1 and section 2 took 5-10 mins to run when theta but this may vary depending on n and theta values you put
# 3) section 3 will take the most amount of time, on my system it took around ~25 mins again depending on the parameters you put

# -------------------------------   #


if __name__ == "__main__":

    # -------------------------------------------------#

    #                  VARIATION WITH theta

    # --------------------------------------------------#

    # m = 5
    # iter = []  # list to store at which epoch , we break the loop to get final new beta
    # costf = []  # final cost functions list

    # for n in range(50, 2000, 400):
    #     betacos_similarity = []
    #     Ycos_similarity = []
    #     for theta in np.linspace(0, 1, 25):  # Variation with theta, 0<=theta<=1
    #         print(f"For n {n} and theta {theta}")
    #         X, Y_true, beta_true = q1.logisticregr(theta, n, m)
    #         beta_new, new_cost, iteration, cosinesim_Y = q2.gradientdescent(X, Y_true, 10000, 0.000001, 0.001)
    #         Ycos_similarity.append(cosinesim_Y)

    #     plt.plot(np.linspace(0, 1, 25), Ycos_similarity, marker="*", label="Y vector, n = " + str(n))
    # plt.ylabel("cosine similarity")
    # plt.xlabel("theta")
    # plt.title("Variation of cosine similarity")
    # plt.legend()
    # plt.show()

    # -------------------------------------------------#

    #                  VARIATION WITH N

    # --------------------------------------------------#
    # m = 5

    # for theta in np.linspace(0, 1, 4):
    #     betacos_similarity = []
    #     Ycos_similarity = []
    #     for n in range(100, 2000, 400):  # Variation with n
    #         print(f"For theta {theta} and n {n}")
    #         X, Y_true, beta_true = q1.logisticregr(theta, n, m)
    #         beta_new, new_cost, iteration, cosinesim_Y = q2.gradientdescent(X, Y_true, 10000, 0.000001, 0.001)
    #         # betacos_similarity.append(cosinesim_beta)
    #         Ycos_similarity.append(cosinesim_Y)

    #     # plt.plot(range(10, 2000, 100), betacos_similarity, marker="*", label="Beta vector, theta = " + str(theta))
    #     plt.plot(range(100, 2000, 400), Ycos_similarity, marker="*", label="Y vector, theta = " + str(round(theta, 2)))
    # plt.xlabel("n")
    # plt.ylabel("cosine similarity")
    # plt.title("Variation of cosine similarity")
    # plt.legend()
    # plt.show()

    # -------------------------------------------------#

    #         VARIATION WITH BOTH theta and N

    # --------------------------------------------------#

    ninputs = np.concatenate((np.linspace(10, 1999, 10, dtype=int), np.linspace(2000, 20000, 10, dtype=int))) # non-linear spacing
    theta = np.linspace(0.01, 0.99, 10)
    cosinesim_Y = np.zeros((10, 20))
    cosinesim_beta = np.zeros((10, 20))
    m = 10
    for t in theta:
        for n in ninputs:
            print(f"For n {n} and theta {t}")
            X, Y_true, beta_true = q1.logisticregr(t, n, m)
            (
                beta_new,
                new_cost,
                iteration,
                cosinesim_Y[np.where(theta == t)[0], np.where(ninputs == n)[0]],
            ) = q2.gradientdescent(X, Y_true, 10000, 0.000001, 0.001)

    plt.title("Cosine Similarity (Y vector)")
    plt.contourf(ninputs, theta, cosinesim_Y, cmap="inferno")
    c = plt.colorbar()
    c.set_label("cosine similarity")
    plt.xlabel("n")
    plt.ylabel(r"$\theta$")
    plt.show()
