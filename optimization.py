import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from systems import ToySystem

from settings import interval, frames


if __name__ == "__main__":
    system_str = 'quadrotor'

    plt.close('all')
    toy = ToySystem(system_str, draw=False)

    from control import lyap

    A, B, Q, R, S = [toy.lqr_data[key] for key in ['A', 'B', 'Q', 'R', 'S']]
    n, m = B.shape
    X0 = np.eye(n)

    Kare = toy.K

    def policy_cost(K):
        AK = A+np.dot(B, K)
        if np.any(np.real(la.eig(AK)[0]) > 0):
            return np.inf
        else:
            QRS = np.block([[Q, S],
                            [S.T, R]])
            IK = np.vstack([np.eye(n), K])
            QK = np.dot(IK.T, np.dot(QRS, IK))

            X = lyap(AK.T, X0)
            return np.trace(np.dot(QK, X))

    def policy_cost_gradient(K):
        AK = A+np.dot(B, K)
        QRS = np.block([[Q, S],
                        [S.T, R]])
        IK = np.vstack([np.eye(n), K])
        QK = np.dot(IK.T, np.dot(QRS, IK))

        P = lyap(AK, QK)
        X = lyap(AK.T, X0)
        return 2*np.dot((np.dot(R, K) + np.dot(B.T, P) + S.T), X)


    ng = 100
    idx1, idx2 = [1, 0], [1, 5]
    Kare1, Kare2 = Kare[idx1[0], idx1[1]], Kare[idx2[0], idx2[1]]
    b_lwr, b_upr = 0, 4
    K1pts, K2pts = np.linspace(b_lwr*Kare1, b_upr*Kare1, ng), np.linspace(b_lwr*Kare2, b_upr*Kare2, ng)
    K1mesh, K2mesh = np.meshgrid(K1pts, K2pts)

    Cmesh = np.zeros([ng, ng])

    for i in range(ng):
        for j in range(ng):
            K = np.copy(Kare)
            K[idx1[0], idx1[1]] = K1mesh[i, j]
            K[idx2[0], idx2[1]] = K2mesh[i, j]

            Cmesh[i, j] = policy_cost(K)

    c_min = policy_cost(Kare)
    c_max = 10*c_min
    for i in range(ng):
        for j in range(ng):
            if Cmesh[i, j] > c_max:
                Cmesh[i, j] = np.inf

    plt.figure()
    plt.contour(K1mesh, K2mesh, Cmesh, levels=np.linspace(c_min, c_max, 30), zorder=1)

    K = np.copy(Kare)
    # K[idx1[0], idx1[1]] = 0.4*Kare1
    # K[idx2[0], idx2[1]] = 3.5*Kare2
    K[idx1[0], idx1[1]] = 0.6*Kare1
    K[idx2[0], idx2[1]] = 2.0*Kare2

    plt.scatter(K[idx1[0], idx1[1]], K[idx2[0], idx2[1]], c='r', zorder=100)
    plt.scatter(Kare[idx1[0], idx1[1]], Kare[idx2[0], idx2[1]], c='k', marker='*', zorder=100)

    toy.K = K

    from time import sleep

    toy.init_plot()
    anim = ani.FuncAnimation(toy.fig, toy.update, interval=interval, blit=True)
    plt.show()
    sleep(1)
    plt.close(toy.fig)

    num_steps = 1000
    learning_rate = 0.001
    for i in range(num_steps):
        print(policy_cost(K))
        G = policy_cost_gradient(K)

        # Full update
        K -= learning_rate*G

        # # Projected partial update
        # K[idx1[0], idx1[1]] -= 0.0001*G[idx1[0], idx1[1]]
        # K[idx2[0], idx2[1]] -= 0.001*G[idx2[0], idx2[1]]

        color = np.array([[1 - i/num_steps, 0, 0]])
        plt.scatter(K[idx1[0], idx1[1]], K[idx2[0], idx2[1]], c=color, zorder=100)

    toy.K = K
    print('')

    print(policy_cost(Kare))
    # toy.K = Kare

    toy.init_plot()
    anim = ani.FuncAnimation(toy.fig, toy.update, interval=interval, blit=True)
    plt.show()
