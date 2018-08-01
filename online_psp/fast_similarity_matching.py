# Title: fast_similarity_matching.py
# Description: A function for principal subspace projection using the Hebbian/anti-Hebbian minimax algorithm of
#              Pehlevan et al. with added Sherman-Morrison formula
# Author: Victor Minden (vminden@flatironinstitute.org)and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Reference: submitted

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


class FSM:
    """
    Parameters:
    ====================
    K             -- Dimension of PCA subspace to learn
    D             -- Dimensionality of data
    Minv0         -- Initial guess for the inverse lateral weight matrix M, must be of size K-by-K
    W0            -- Initial guess for the forward weight matrix W, must be of size K-by-D
    learning_rate -- Learning rate as a function of t
    tau           -- Learning rate factor for M (multiplier of the W learning rate)

    Methods:
    ========
    fit_next()

    """

    def __init__(self, K, D, Minv0=None, W0=None, learning_rate=eta, tau=0.5):

        if Minv0 is not None:
            assert Minv0.shape == (K, K), "The shape of the initial guess Minv0 must be (K,K)=(%d,%d)" % (K, K)
            Minv = Minv0
        else:
            Minv = np.eye(K)

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
        self.Minv = Minv
        self.W = W

        # variable to allocate memory and optimize outer product
        self.outer_W = np.empty_like(W)
        self.outer_Minv = np.empty_like(Minv)

    def fit_next(self, x):

        assert x.shape == (self.D,)

        t, tau, W, Minv, K = self.t, self.tau, self.W, self.Minv, self.K

        y = np.dot(Minv, W.dot(x))

        # Plasticity, using gradient ascent/descent
        # TODO: probably factor of 2 can go
        # W <- W + 2 eta(t) * (y*x' - W)
        step = self.eta(t)

        np.outer(2 * step * y, x, self.outer_W)
        W = (1 - 2 * step) * W + self.outer_W

        # M <- M + eta(self.t)/tau * (y*y' - M), using SMW
        step = step / tau
        Minv = Minv / (1 - step)
        z = Minv.dot(y)
        c = step / (1 + step * np.dot(z, y))
        np.outer(c * z, z.T, self.outer_Minv)
        Minv = Minv - self.outer_Minv

        self.Minv = Minv
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

        components = np.asarray(self.Minv.dot(self.W).T)
        if orthogonalize:
            components, _ = np.linalg.qr(components)

        return components
