import sys
import argparse
from scipy.misc import imread, imsave
import numpy as np
import time
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
import sobol
from scipy import stats

import matplotlib.pyplot as plt


def gaussian_kernel(data, sigma, max_distance):
    """Compute the gaussian kernel matrix.

    :param data: data matrix
    :param sigma: parameter sigma of the gaussian kernel
    :return: gaussian kernel matrix
    """
    assert len(data.shape) == 2
    assert sigma > 0

    factor = -0.5 /  (sigma ** 2)
    limit = np.exp(factor*max_distance**2)
    # Find the pairwise squared distances and compute the Gaussian kernel.
    K = []
    for k in data:
        d = np.exp(factor*np.sum((data - k)**2,axis=1))
        d[d < limit] = 0.0  # truncate the Gaussian
        d = scipy.sparse.csc_matrix(d[:,None])
        K.append(d)
    K = scipy.sparse.hstack(K)
    return K


def exp_kernel(data, gamma, rho, max_distance):
    """Compute the gaussian kernel matrix.

    :param data: data matrix
    :param sigma: parameter sigma of the gaussian kernel
    :return: gaussian kernel matrix
    """
    assert len(data.shape) == 2

    limit = np.exp(-(max_distance / rho)**gamma)
    # Find the pairwise squared distances and compute the Exponential kernel.
    K = []
    for k in data:
        d = np.exp(-np.sum((np.abs(data - k)/rho)**gamma, axis=1))
        d[d < limit] = 0.0  # truncate the Gaussian
        d = scipy.sparse.csc_matrix(d[:, None])
        K.append(d)
    K = scipy.sparse.hstack(K)
    return K


def matern_kernel(P, sigma_rho=4, sigma_gamma=1, sigma_tau=1):

    P_arr = np.asarray(P)
    rhos   = P_arr[:, 0]
    gammas = P_arr[:, 1]
    taus   = P_arr[:, 2]

    t = rhos.shape[0]
    M = np.zeros((t, t))

    for i, (rho, gamma, tau) in enumerate(zip(rhos, gammas, taus)):

        s_sq = ((rhos - rho)/sigma_rho)**2 + \
               ((gammas - gamma)/sigma_gamma)**2 + \
               ((taus - tau)/sigma_tau)**2

        M[i, :] = (1 + np.sqrt(5*s_sq) + 5./3*s_sq)*np.exp(-np.sqrt(5*s_sq))

    return M


class KernelRidgeRegressor(object):
    """Kernel Ridge Regressor.
    """

    def __init__(self, tau=0.8, sigma=4.0, gamma=1.8, rho=7.5, kernel_fct='Gaussian'):
        self.dim = None
        self.train_x = None
        self.alpha = None
        self.mean_y = None
        self.std_y = None

        self.tau = tau
        self.sigma = sigma
        self.gamma = gamma
        self.rho = rho
        self.scale = -0.5 / sigma**2
        self.max_distance = 4.0*sigma

        self.kernel_fct = kernel_fct

    def compute_alpha(self, train_x, train_y):
        """Compute the alpha vector of the ridge regressor.

        :param train_x: training x data
        :param train_y: training y data
        :param tau: parameter tau of the ridge regressor
        :param sigma: parameter sigma of the gaussian kernel
        :return: alpha vector
        """
        #print "building input kernel matrix"
        if self.kernel_fct == 'Gaussian':
            K = gaussian_kernel(train_x, self.sigma, self.max_distance)
        elif self.kernel_fct == 'Exponential':
            K = exp_kernel(train_x, self.gamma, self.rho,  self.max_distance)
        else:
            print 'Invalid kernel fct!'
            sys.exit(0)

        print "sparsity: %.2f%%" % (float(100*K.nnz) / (K.shape[0]*K.shape[1]))
        M = K + self.tau * scipy.sparse.identity(train_x.shape[0])
        y = scipy.sparse.csc_matrix(train_y[:, None])
        #print "solving sparse system"
        alpha = scipy.sparse.linalg.cg(M, train_y)
        #print "done computing alpha"
        return alpha[0]

    def train(self, train_x, train_y):
        """Train the kernel ridge regressor.

        :param train_x: training x data
        :param train_y: training y data
        """
        assert len(train_x.shape) == 2
        assert len(train_y.shape) == 1
        assert train_x.shape[0] == train_y.shape[0]

        self.dim = train_x.shape[1]
        self.train_x = train_x.astype(np.float32)
        self.tree = scipy.spatial.cKDTree(self.train_x)

        self.mean_y = train_y.mean()
        self.std_y = train_y.std()
        train_y_std = (train_y - self.mean_y) / self.std_y

        self.alpha = self.compute_alpha(self.train_x, train_y_std)

    def predict_single(self, pred_x):
        """Predict the value of a single instance.

        :param pred_x: x data
        :return: predicted value of pred_x
        """
        assert len(pred_x.shape) == 1
        assert pred_x.shape[0] == self.dim
        indices = np.asarray(self.tree.query_ball_point(pred_x, self.max_distance))
        dist = np.sum((self.train_x[indices]-pred_x)**2, axis=1)
        kappa = np.exp(self.scale*dist)
        pred_y = np.dot(kappa, self.alpha[indices])
        return self.std_y * pred_y + self.mean_y

    def predict(self, pred_x):
        """Predict the values of pred_x.

        :param pred_x: x data
        :return: predicted values of pred_x
        """
        assert len(pred_x.shape) == 2
        assert pred_x.shape[1] == self.dim
        pred_x = pred_x.astype(np.float32)
        return np.array([self.predict_single(x) for x in pred_x])


def kernel_ridge_regression(**kwargs):

    # Load the image.
    im_orig = np.squeeze(imread("cc_90.png"))

    # Make a copy, so both the original and the regressed image can be shown afterwards.
    im = np.array(im_orig)

    # Find the known pixels and the pixels that shall be predicted.
    known_ind = np.where(im != 0)
    unknown_ind = np.where(im >= 0)
    known_x = np.array(known_ind).transpose()
    known_y = np.array(im[known_ind])
    pred_x = np.array(unknown_ind).transpose()

    # Train and predict with the given regressor.
    r = KernelRidgeRegressor(**kwargs)
    r.train(known_x, known_y)
    pred_y = r.predict(pred_x)

    # Write the predicted values back into the image and show the result.
    im[unknown_ind] = pred_y

    return im


def get_m(P, theta, sigma_rho=4, sigma_gamma=1, sigma_tau=1):

    P_arr = np.asarray(P)
    rhos   = P_arr[:, 0]
    gammas = P_arr[:, 1]
    taus   = P_arr[:, 2]

    t = rhos.shape[0]
    m = np.zeros((t, 1))

    rho = theta[0]
    gamma = theta[1]
    tau = theta[2]

    s_sq = ((rhos - rho)/sigma_rho)**2 + \
           ((gammas - gamma)/sigma_gamma)**2 + \
           ((taus - tau)/sigma_tau)**2

    m[:, 0] = (1 + np.sqrt(5*s_sq) + 5./3*s_sq)*np.exp(-np.sqrt(5*s_sq))
    return m


def compute_mse(image_pred, image_gt):

    assert image_pred.shape == image_gt.shape

    h = image_gt.shape[0]
    w = image_gt.shape[1]

    mse = 1./(h*w) * np.sum((image_pred - image_gt)**2)

    return mse


def compute_msea(E, M, m, _lambda=0.3):
    M_reg = M + _lambda*np.diag(np.ones((M.shape[0])))
    M_reg_m = scipy.linalg.solve(M_reg, m)
    msea = np.asarray(E).T.dot(M_reg_m)

    m_star = 1.  # for all theta
    msea_var = m_star - m.T.dot(M_reg_m)

    return msea, msea_var


def main():
    """."""

    im_gt = np.squeeze(imread('charlie-chaplin.jpg'))

    rho_limits = [0.01, 10]
    gamma_limits = [0.01, 2]
    tau_limits = [0.01, 10]

    param_lb = np.array([rho_limits[0], gamma_limits[0], tau_limits[0]])
    param_ub = np.array([rho_limits[1], gamma_limits[1], tau_limits[1]])

    sample_size = 2000

    Q = np.zeros((sample_size, 3))

    for i in range(sample_size):
        rho, gamma, tau = sobol.i4_sobol(3, i)[0] * (param_ub - param_lb) + param_lb

        Q[i, 0] = rho
        Q[i, 1] = gamma
        Q[i, 2] = tau

    P = []
    E = []

    # Get initial set of parameters P and corresponding MSEs E
    for i in range(20):
        print i
        P.append(np.array([Q[i, 0], Q[i, 1], Q[i, 2]]))
        im_pred = kernel_ridge_regression(
            kernel_fct='Exponential', rho=Q[i, 0], gamma=Q[i, 1], tau=Q[i, 2])
        mse = compute_mse(im_pred, im_gt)
        E.append(mse)

    E_best = np.min(E)

    # Iterate improving hyperparameters
    for j in range(10):
        print 'Iteration hyperparameter improvement: {}'.format(j)
        u_best = -np.inf
        theta_best = None
        for i in range(sample_size):

            M = matern_kernel(P)

            theta = np.array([Q[i, 0], Q[i, 1], Q[i, 2]])
            m = get_m(P, theta)

            E_i, var_msea = compute_msea(E, M, m)

            gamma = (E_best - E_i) / np.sqrt(var_msea)

            u = np.sqrt(var_msea)*(gamma*stats.norm.cdf(gamma) +
                                   stats.norm.pdf(gamma))
            if u > u_best:
                u_best = u
                theta_best = theta.copy()

        P.append(theta_best)

        im_pred = kernel_ridge_regression(kernel_fct='Exponential',
            rho=theta_best[0], gamma=theta_best[1], tau=theta_best[2])
        mse = compute_mse(im_pred, im_gt)
        E.append(mse)

        if mse < E_best:
            E_best = mse
            print 'E_best improved: {}'.format(E_best)

    return 0


if __name__ == "__main__":
    status = main()
    sys.exit(status)
