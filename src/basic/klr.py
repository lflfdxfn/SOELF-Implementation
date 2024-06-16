import numpy as np

from scipy.special import expit
from numpy.linalg import norm

class OnlineKLR:

    def __init__(self, eta, lamda, t, cnt, n_dim):
        # hyper-parameters for KLR
        self.eta = eta
        self.lamda = lamda
        self.t = t
        self.cnt = cnt
        self.n_dim = n_dim

        # initialize an empty model
        self.currentAlpha = np.zeros([1, self.cnt], dtype = float)
        self.norm2X = np.zeros([1, self.cnt], dtype = float)
        self.trainFea = np.zeros([self.n_dim, self.cnt], dtype = float)
        self.index = 0
        self.firstloop = 1
        self.active = 1

    # copy
    def copy(self):
        new_OnlineKLR = OnlineKLR(self.eta, self.lamda, self.t, self.cnt, self.n_dim)
        new_OnlineKLR.currentAlpha = self.currentAlpha.copy()
        new_OnlineKLR.norm2X = self.norm2X.copy()
        new_OnlineKLR.trainFea = self.trainFea.copy()
        new_OnlineKLR.index = self.index
        new_OnlineKLR.firstloop = self.firstloop
        new_OnlineKLR.active = self.active

        return new_OnlineKLR

    # classify
    def classify(self, xt):
        # the shape of xt:
        assert xt.shape == (self.n_dim, 1)

        # length of used vectors
        if (self.firstloop == 1):
            T = self.index
        else:
            T = self.cnt

        sigma = self.t
        norm2xt = norm(xt)**2

        # Depends on the kernel
        xtTrainFea = (xt.T @ self.trainFea[:, 0:T])
        k_xt = self.construct_RBF_Row(norm2xt, self.norm2X[[0], 0:T], xtTrainFea, sigma)
        ft_xt = (k_xt @ (self.currentAlpha[[0], 0:T].T)) [0, 0]
        prob_yt_ft = expit(ft_xt)

        return prob_yt_ft, ft_xt

    # update
    def update(self, xt, yt, ft_xt):
        # the shape of xt, yt, ft_xt
        assert xt.shape == (self.n_dim, 1)
        assert (yt == 1) or (yt == -1)

        # Columns are samples
        # Optimized for Gaussian Kernel
        if (self.firstloop == 1):
            T = self.index
        else:
            T = self.cnt

        sigma = self.t

        norm2xt = norm(xt)**2

        # gradient descent updation
        prob_yt_ft = expit(ft_xt * yt)

        tmp = self.eta * yt * (1 - prob_yt_ft)
        param = (1 - self.eta * self.lamda)
        new_alpha = tmp
        new_norm = norm2xt

        return param, new_alpha, new_norm

    # RBF vector
    def construct_RBF_Row(self, norm2xt, norm2X, xtTrainFea, sigma):
        xtx = norm2X + norm2xt - 2 * xtTrainFea
        k_xt = np.exp(xtx/(-2 * sigma ** 2))

        return k_xt


class Ensemble:
    def __init__(self, model:OnlineKLR, n_ensemble:int, args = (), kwds = dict()):
        # parameters
        self.model = model
        self.m = n_ensemble
        self.model_args = args
        self.model_kwds = kwds
        self.active = 1

        # initial models
        self.model_list = [];

        for i in range(n_ensemble):
            self.model_list.append(model(*self.model_args, **self.model_kwds))

    def copy(self):
        new_Ensemble = Ensemble(self.model, self.m, self.model_args, self.model_kwds)
        new_Ensemble.active = self.active
        new_Ensemble.model_list = []
        for model in self.model_list:
            new_Ensemble.model_list.append(model.copy())

        return new_Ensemble

    def classify(self, xt):
        yts = []

        for model in self.model_list:
            yts.append(model.classify(xt))

        return list(zip(*yts))









