# Title: similarity_matching.py
# Description: A function for principal subspace projection using the Hebbian/anti-Hebbian minimax algorithm of Pehlevan et al.
# Author: Victor Minden (vminden@flatironinstitute.org)and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Reference: C. Pehlevan, A. M. Sengupta, and D. B. Chklovskii, “Why do similarity matching objectives lead to
#            Hebbian/anti-Hebbian networks?”. Neural Computation, vol. 30, no. 1, pp. 84–124, 2018

##############################
# Imports
import numpy as np

##############################


def eta(t):
    """
    Parameters:
    ====================
    t -- time at which learning rate is to be evaluated

    Output:
    ====================
    step -- learning rate at time t
    """

    return 1.0 / (t + 5)


class SM:
    """
    Parameters:
    ====================
    K             -- Dimension of PCA subspace to learn
    D             -- Dimensionality of data
    M0            -- Initial guess for the lateral weight matrix M, must be of size K-by-K
    W0            -- Initial guess for the forward weight matrix W, must be of size K-by-D
    learning_rate -- Learning rate as a function of t
    tau           -- Learning rate factor for M (multiplier of the W learning rate)

    Methods:
    ========
    fit_next()

    """

    def __init__(self, K, D, M0=None, W0=None, learning_rate=eta, tau=0.5):

        if M0 is not None:
            assert M0.shape == (K, K), "The shape of the initial guess Minv0 must be (K,K)=(%d,%d)" % (K, K)
            M = M0
        else:
            M = np.eye(K)

        if W0 is not None:
            assert W0.shape == (K, D), "The shape of the initial guess W0 must be (K,D)=(%d,%d)" % (K, D)
            W = W0
        else:
            W = np.random.normal(0, 1.0 / np.sqrt(D), size=(K, D))


        self.eta = learning_rate
        self.t = 0

        self.K = K
        self.D = D
        self.tau = tau
        self.M = M
        self.W = W

        # Storage variables to allocate memory and optimize outer product time
        self.outer_W = np.empty_like(W)
        self.outer_M = np.empty_like(M)

    def fit_next(self, x):

        assert x.shape == (self.D,)

        t, tau, W, M, K = self.t, self.tau, self.W, self.M, self.K

        y = np.linalg.solve(M, W.dot(x))

        # Plasticity, using gradient ascent/descent
        # TODO: the factor of 2 can go away probably...
        # W <- W + 2 eta(t) * (y*x' - W)
        step = self.eta(t)

        np.outer(2 * step * y, x, self.outer_W)
        W = (1 - 2 * step) * W + self.outer_W

        # M <- M + eta(self.t)/tau * (y*y' - M)
        step = step / tau
        np.outer(step * y, y, self.outer_M)
        M = (1 - step) * M + self.outer_M

        self.M = M
        self.W = W

        self.t += 1

    def get_components(self, orthogonalize=True):
        '''
        Extract components from object

        orthogonalize: bool
            whether to orthogonalize the components before returning

        Returns
        -------
        components: ndarray
        '''

        components = np.asarray(np.linalg.solve(self.M, self.W).T)
        if orthogonalize:
            components, _ = np.linalg.qr(components)

        return components

