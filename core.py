import math
import scipy
import scipy.linalg
import torch.linalg
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

import random
import bisect
from torch import Tensor

PI = 3.141592653589793
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


class Array:
    """
    A class to hold the data and meta information
    """

    def __init__(self, arr=None, mu: float = 0.0, std: float = 1.0, dtype='numpy'):
        assert dtype in ('numpy', 'torch'), f'dtype={dtype} must be "numpy" or "torch"'
        tmpflag = arr is not None
        if tmpflag and dtype == 'numpy':
            assert isinstance(arr, np.ndarray)
        if tmpflag and dtype == 'torch':
            assert isinstance(arr, torch.Tensor)
        self.arr = arr
        self.mu = mu
        self.std = std
        self.dtype = dtype

    def __len__(self):
        return len(self.arr)

    # self.arr is the result of normalizing the raw data using self.mu and self.std
    def norm(self, mu=None, std=None):
        if not mu or not std:
            mu = self.arr.mean()
            std = self.arr.std()

        self.arr = (self.arr - mu) / std
        self.mu = self.mu + mu * self.std
        self.std = self.std * std

    # reverse operation of norm
    def un_norm(self, mu=None, std=None):
        if not mu or not std:
            mu, std = self.mu, self.std
        self.arr = self.arr * std + mu
        self.mu = self.mu - mu * self.std / std
        self.std = self.std / std

    # un_norm and center the data
    def center(self):
        self.un_norm(0, self.std)

    # to tensor
    def to_tensor(self):
        if self.dtype == 'torch':
            return self
        return Array(torch.from_numpy(self.arr), float(self.mu), float(self.std), dtype='torch')

    # to numpy
    def to_numpy(self):
        if self.dtype == 'numpy':
            return self
        return Array(self.arr.cpu().detach().numpy(), float(self.mu), float(self.std), dtype='numpy')

    def __iter__(self):
        return self.arr.__iter__()

    def __getitem__(self, item):
        return self.arr.__getitem__(item)

    def __setitem__(self, key, value):
        self.arr[key] = value

    def __str__(self):
        return self.arr.__str__()

    def __add__(self, other):
        assert self.dtype == other.dtype
        cat = get_mat_op('cat', self.dtype)
        if self.arr is None:
            new_arr = other.arr
        else:
            new_arr = cat((self.arr, other.arr), axis=0)  # works for both array and n x 1 matrix
        return Array(new_arr, self.mu, self.std, self.dtype)


class Parameters:
    """
    A container class for holding variables
    """

    def __init__(self, dtype='numpy', default_para_list=['p', 'l', 'sigma_f2', 'sigma_n2', 'ws', 'lb', 'bic']):
        assert dtype in ('numpy', 'torch'), f'dtype={dtype} must be "numpy" or "torch"'
        self.dtype = dtype
        self.para_list = default_para_list

    def disp(self, title='para info', para_list=None):
        if not para_list:
            para_list = self.para_list
        print('-' * 10, title, '-' * 10)
        for e in para_list:
            if not hasattr(self, e):
                continue
            if isinstance(self.__getattribute__(e), torch.Tensor):
                value = np.around(self.__getattribute__(e).detach().numpy(), 2)
            else:
                value = np.around(self.__getattribute__(e), 2)
            print(f'{e}={value}')
    
    def disp_comp(self, title='para info', para_list=None):
        i = 0
        self.disp()
        print('-' * 20)
        for multi_comp in self.component_para_list:
            print(f'component {i}')
            for e in ['ws', 'p', 'l', 'sigma_f2', 'sigma_n2']:
                if not hasattr(multi_comp, e):
                    continue
                if isinstance(multi_comp.__getattribute__(e), torch.Tensor):
                    value = np.around(multi_comp.__getattribute__(e).detach().numpy(), 2)
                else:
                    value = np.around(multi_comp.__getattribute__(e), 2)
                print(f'{e}={value}')
            i = i+1
            print('-' * 10)

    # to tensor
    def to_tensor(self):
        if self.dtype == 'torch':
            return self
        for e in self.__dict__:
            self.__dict__[e] = torch.from_numpy(self.__dict__[e])

    # to numpy
    def to_numpy(self):
        if self.dtype == 'numpy':
            return self
        for e in self.__dict__:
            self.__dict__[e] = self.__dict__[e].cpu().detach().numpy()


def se_kernel(x_arr: Array, y_arr: Array, para: Parameters):
    """
    calculate the SE kernel between x and y
    :param x_arr: N x 1 array, represent rows in covariance matrix
    :param y_arr: M x 1 array, represent columns in covariance matrix
    :param para: parameters of SE kernel: l, sigma_f2
    :return: M x N kernel matrix
    """

    l, sigma2 = para.l, para.sigma_f2
    assert x_arr.dtype == y_arr.dtype
    exp = get_mat_op('exp', x_arr.dtype)
    return sigma2 * exp((x_arr.arr.reshape((-1, 1)) - y_arr.arr.reshape((1, -1))) ** 2 / (-2 * l ** 2))
    # if x_arr.dtype == 'numpy':
    #     return sigma2 * np.exp((x_arr.arr.reshape((-1, 1)) - y_arr.arr.reshape((1, -1))) ** 2 / (-2 * l ** 2))
    # elif x_arr.dtype == 'torch':
    #     return sigma2 * torch.exp((x_arr.arr.reshape((-1, 1)) - y_arr.arr.reshape((1, -1))) ** 2 / (-2 * l ** 2))
    # else:
    #     raise Exception(f'Unknown x_arr.dtype={x_arr.dtype}, must be "numpy" or "torch"')


def pse_kernel(x_arr: Array, y_arr: Array, para: Parameters):
    """
    calculate the periodic SE kernel between x and y
    :param x_arr: N x 1 array, represent rows in covariance matrix
    :param y_arr: M x 1 array, represent columns in covariance matrix
    :param para: parameters of SE kernel: p, l, sigma_f2
    :return: M x N kernel matrix
    """

    p, l, sigma2 = para.p, para.l, para.sigma_f2
    assert x_arr.dtype == y_arr.dtype
    exp = get_mat_op('exp', x_arr.dtype)
    sin = get_mat_op('sin', x_arr.dtype)
    #return sigma2 * exp(-2 * sin(PI * abs(x_arr.arr.reshape((-1, 1)) - y_arr.arr.reshape((1, -1))) / p) ** 2 / (l ** 2)) # testing different funcs
    return sigma2 * exp(- (2 * (np.sin(PI *  abs(x_arr.arr.reshape((-1, 1)) - y_arr.arr.reshape((1, -1))) / p))**2)/ (l**2))

    #return sigma2 * exp(-2 * sin(PI * abs(x_arr.arr.reshape((-1, 1)) - y_arr.arr.reshape((1, -1))) / p) ** 2 / (l ** 2))

    # if x_arr.dtype == 'numpy':
    #     return sigma2 * np.exp((x_arr.arr.reshape((-1, 1)) - y_arr.arr.reshape((1, -1))) ** 2 / (-2 * l ** 2))
    # elif x_arr.dtype == 'torch':
    #     return sigma2 * torch.exp((x_arr.arr.reshape((-1, 1)) - y_arr.arr.reshape((1, -1))) ** 2 / (-2 * l ** 2))
    # else:
    #     raise Exception(f'Unknown x_arr.dtype={x_arr.dtype}, must be "numpy" or "torch"')


def cal_se_cov_mat(x_arr: Array, para: Parameters):
    """
    calculate the covariance matrix for training data x_arr
    :param x_arr: N x 1 array
    :param para: parameters for calculating the covariance matrix
        l: length scale of signal
        sigma_f2: variance of signal
        sigma_n2: variance of noise
    :return: c_mat, SE covariance matrix for x_arr given the parameters
    """
    l, sigma_f2, sigma_n2 = para.l, para.sigma_f2, para.sigma_n2

    mat = se_kernel(x_arr, x_arr, para)

    if x_arr.dtype == 'numpy':
        return mat + sigma_n2 * np.eye(len(x_arr))
    elif x_arr.dtype == 'torch':
        return mat + sigma_n2 * torch.eye(len(x_arr))
    else:
        raise Exception(f'Unknown x_arr.dtype={x_arr.dtype}, must be "numpy" or "torch"')


def cal_pse_cov_mat(x_arr: Array, para: Parameters):
    """
    calculate the covariance matrix for training data x_arr
    :param x_arr: N x 1 array
    :param para: parameters for calculating the covariance matrix
        p: period
        l: length scale of signal
        sigma_f2: variance of signal
        sigma_n2: variance of noise
    :return: c_mat, SE covariance matrix for x_arr given the parameters
    """
    p, l, sigma_f2, sigma_n2 = para.p, para.l, para.sigma_f2, para.sigma_n2

    mat = pse_kernel(x_arr, x_arr, para)

    if x_arr.dtype == 'numpy':
        return mat + sigma_n2 * np.eye(len(x_arr))
    elif x_arr.dtype == 'torch':
        return mat + sigma_n2 * torch.eye(len(x_arr))
    else:
        raise Exception(f'Unknown x_arr.dtype={x_arr.dtype}, must be "numpy" or "torch"')


def np_logdet(x):
    s, v = np.linalg.slogdet(x)
    return v


# get matrix operators for given dtype
def get_mat_op(op_name, dtype='torch'):
    op_name_list = ('matmul', 'inverse', 'det', 'logdet', 'log', 'exp', 'sqrt',
                    'cholesky', 'cat', 'sin', 'randn', 'diagonal', 'searchsorted')
    assert op_name in op_name_list, \
        f'op_name={op_name} must be [{".".join(op_name_list)}]'
    d = dict()
    if dtype == 'numpy':
        d['matmul'] = np.matmul
        d['inverse'] = np.linalg.inv
        d['det'] = np.linalg.det
        d['logdet'] = np_logdet
        d['log'] = np.log
        d['exp'] = np.exp
        d['sqrt'] = np.sqrt
        d['cholesky'] = scipy.linalg.cholesky
        d['cat'] = np.concatenate
        d['sin'] = np.sin
        d['randn'] = np.random.randn
        d['diagonal'] = np.diagonal
        d['searchsorted'] = np.searchsorted
    elif dtype == 'torch':
        d['matmul'] = torch.mm
        d['inverse'] = torch.inverse
        d['det'] = torch.linalg.det
        d['logdet'] = torch.logdet
        d['log'] = torch.log
        d['exp'] = torch.exp
        d['sqrt'] = torch.sqrt
        #d['cholesky'] = torch.cholesky
        d['cholesky'] = torch.linalg.cholesky
        d['cat'] = torch.cat
        d['sin'] = torch.sin
        d['randn'] = torch.randn
        d['diagonal'] = torch.diagonal
        d['searchsorted'] = torch.searchsorted
    else:
        raise Exception(f'Unknown x_arr.dtype={type}, must be "numpy" or "torch"')
    return d[op_name]


def mvn_log_pdf(x_arr: Array, mu, sigma, inv_sigma=None):
    """
    calculate the log pdf of multivariate normal
    :param x_arr: multi-dimensional data
    :param mu: mean vector
    :param sigma: covariance matrix
    :param inv_sigma: inverse of covariance matrix
    :return: log pdf
    """
    if (isinstance(mu, int) or isinstance(mu, float)) and mu < 1e-6:
        assert len(x_arr) == len(sigma)
    else:
        assert len(x_arr) == len(mu) == len(sigma)
        assert x_arr.arr.shape == mu.shape

    n = len(x_arr)

    matmul = get_mat_op('matmul', x_arr.dtype)
    inverse = get_mat_op('inverse', x_arr.dtype)
    # det = get_mat_op('det', x_arr.dtype)
    logdet = get_mat_op('logdet', x_arr.dtype)
    if  x_arr.dtype == 'torch':
        tmparr = (x_arr.arr.float() - mu).reshape((-1, 1))  # n x 1
    else:
        tmparr = (x_arr.arr - mu).reshape((-1, 1))  # n x 1
        
    if inv_sigma is None:
        tmp_mat = matmul(tmparr.T, inverse(sigma))
    else:
        tmp_mat = matmul(tmparr.T, inv_sigma)
    tmp_term2 = matmul(tmp_mat, tmparr)

    # return -0.5 * n * math.log(2 * PI) - 0.5 * math.log(abs(det(sigma))) - 0.5 * tmp_term2
    return -0.5 * n * math.log(2 * PI) - 0.5 * logdet(sigma) - 0.5 * tmp_term2


class GaussianProcess:
    """
    A class for Gaussian process, GP(0, covmat)
    """

    def __init__(self, x_arr: Array = None, y_arr: Array = None, t_arr: Array = None, kernel_para: Parameters = None,
                 ker_func=se_kernel, covmat_func=cal_se_cov_mat, inv_flag=False):
        self.x_arr = x_arr
        self.y_arr = y_arr
        self.t_arr = t_arr
        self.ker_func = ker_func
        self.covmat_func = covmat_func

        self.kernel_para = kernel_para
        self.covmat = None
        self.inv_covmat = None
        self.inv_flag = inv_flag  # flag of whether updating the inverse of the covariance matrix

        if kernel_para:
            self.update_kernel_para(kernel_para)

    def __add__(self, other):
        if self.x_arr is None:
            return GaussianProcess(x_arr=other.x_arr, y_arr=other.y_arr, t_arr=other.t_arr, ker_func=other.ker_func,
                                   covmat_func=other.covmat_func, inv_flag=other.inv_flag)
        new_x_arr = self.x_arr + other.x_arr
        new_t_arr = self.t_arr + other.t_arr
        new_y_arr = None
        if self.y_arr and self.y_arr:
            new_y_arr = self.y_arr + self.y_arr
        return GaussianProcess(x_arr=new_x_arr, y_arr=new_y_arr, t_arr=new_t_arr, ker_func=self.ker_func,
                               covmat_func=self.covmat_func, inv_flag=self.inv_flag)

    # update kernel parameters
    def update_kernel_para(self, kernel_para):
        self.kernel_para = kernel_para
        self.covmat = self.covmat_func(self.x_arr, self.kernel_para)
        if self.inv_flag:
            self.inv_covmat = self.cal_inv_covmat(self.covmat)

    # update noise variance
    def update_kernel_sigma_n2(self, sigma_n2):
        self.kernel_para.sigma_n2 = sigma_n2

    # update the mean of the GP, i.e. y_arr
    def update_mu(self, y_arr):
        self.y_arr = y_arr

    # update covariance matrix of the GP
    def update_covmat(self, covmat):
        self.covmat = covmat

    def cal_inv_covmat(self, covmat):
        inverse = get_mat_op('inverse', self.x_arr.dtype)
        return inverse(covmat)

    # calculate the log marginal likelihood, i.e. p(t|x,\theta) = N(t | 0, covmat)
    def cal_marginal_loglik(self):
        if self.inv_flag:
            return mvn_log_pdf(self.t_arr, 0, self.covmat, self.inv_covmat)
        else:
            return mvn_log_pdf(self.t_arr, 0, self.covmat)

    # calculate the mean and covmat of the predictive distribution, i.e. p(t'|x,t, x', \theta)
    def cal_pred_mu(self, gp):  # gp is an instance of GaussianProcess
        assert self.x_arr.dtype == gp.x_arr.dtype, f'self.x_arr.dtype={self.x_arr.dtype} != x_arr_test.arr.dtype={gp.x_arr.dtype}'

        ker = self.ker_func(self.x_arr, gp.x_arr, self.kernel_para)  # n x nt
        if not self.inv_flag:
            self.inv_covmat = self.cal_inv_covmat(self.covmat)  # n x n
        matmul = get_mat_op('matmul', self.x_arr.dtype)
        tmpmat = matmul(ker.T, self.inv_covmat)  # nt x n x n x n => nt x n
        mu = matmul(tmpmat, self.t_arr.arr)

        return mu

    # calculate the mean and covmat of the predictive distribution, i.e. p(t'|x,t, x', \theta)
    def cal_pred_mu_sigma(self, gp):  # gp is an instance of GaussianProcess
        assert self.x_arr.dtype == gp.x_arr.dtype, f'self.x_arr.dtype={self.x_arr.dtype} != x_arr_test.arr.dtype={gp.x_arr.dtype}'

        ker = self.ker_func(self.x_arr, gp.x_arr, self.kernel_para)  # n x nt
        if not self.inv_flag:
            self.inv_covmat = self.cal_inv_covmat(self.covmat)  # n x n
        matmul = get_mat_op('matmul', self.x_arr.dtype)
        tmpmat = matmul(ker.T, self.inv_covmat)  # nt x n x n x n => nt x n
        if self.x_arr.dtype == 'torch':
            mu = matmul(tmpmat, self.t_arr.arr.float())
        else:
            mu = matmul(tmpmat, self.t_arr.arr)
        # covmat_test = self.covmat_func(gp.x_arr, gp.kernel_para)  # nt x nt
        # sigma = covmat_test - matmul(tmpmat, ker)  # nt x nt
        sigma = gp.covmat - matmul(tmpmat, ker)  # nt x nt, here we use gp.covmat to save some computation
        return mu, sigma

    # calculate the log predictive likelihood, i.e. p(t'|x,t, x', \theta)
    def cal_pred_loglik(self, gp):  # gp is an instance of GaussianProcess
        mu, sigma = self.cal_pred_mu_sigma(gp)
        return mvn_log_pdf(gp.t_arr, mu, sigma)

    # calculate the log predictive likelihood using student t, i.e. p(t'|x,t, x', \theta)
    def cal_pred_t_loglik(self, gp):
        mu = self.cal_pred_mu(gp)
        residual = gp.t_arr.arr - mu
        return torch.sum(
            torch.distributions.studentT.StudentT(
                df=2, loc=0.0, scale=torch.sqrt(gp.kernel_para.sigma_n2), validate_args=None).log_prob(residual))

    # approximate predictive mean using y_arr
    # x_arr must be sorted
    def approx_pred_mu(self, x_arr):
        assert self.y_arr is not None
        searchsorted = get_mat_op('searchsorted', x_arr.dtype)
        inds = searchsorted(self.x_arr.arr, x_arr.arr)
        return self.y_arr.arr[inds]

    def approx_pred_loglik(self, gp):
        mu = self.approx_pred_mu(gp.x_arr)
        residual = gp.t_arr.arr - mu
        if isinstance(gp.kernel_para.sigma_n2, float):
            gp.kernel_para.sigma_n2 = torch.tensor(gp.kernel_para.sigma_n2)

        # return torch.sum(
        #     torch.distributions.studentT.StudentT(
        #         df=1, loc=0.0, scale=torch.sqrt(gp.kernel_para.sigma_n2), validate_args=None).log_prob(residual))

        return torch.sum(
            torch.distributions.normal.Normal(
                loc=0.0, scale=torch.sqrt(gp.kernel_para.sigma_n2), validate_args=None).log_prob(residual))

    # sample a random function
    def sample_f(self):
        if self.covmat is None:
            return
        dtype = self.x_arr.dtype
        randn = get_mat_op('randn', dtype=dtype)
        cholesky = get_mat_op('cholesky', dtype=dtype)
        matmul = get_mat_op('matmul', dtype=dtype)
        u_vec = randn(len(self.x_arr))
        tmp_L_mat = cholesky(self.covmat)  # Tensor
        t_arr = Array(matmul(tmp_L_mat, u_vec.reshape(-1, 1)), dtype=dtype)
        return t_arr

    def plot_f(self, std=None, std_flag=True, color=None):
        mu, sigma = self.cal_pred_mu_sigma(self)

        if torch.is_tensor(mu) and mu.requires_grad:
            mu = mu.detach()
            sigma = sigma.detach()

        diagonal = get_mat_op('diagonal', dtype=self.x_arr.dtype)
        sqrt = get_mat_op('sqrt', dtype=self.x_arr.dtype)

        if std is None:
            std = sqrt(diagonal(sigma))
        x_arr = self.x_arr.arr

        x_arr = x_arr.flatten()
        mu = mu.flatten()

        if color:
            plt.plot(x_arr, mu, '--', color=color)
        else:
            plt.plot(x_arr, mu, '--')

        if std_flag:
            if color:
                plt.fill_between(x_arr, mu - std, mu + std, color=color, alpha=0.5)
            else:
                plt.fill_between(x_arr, mu - std, mu + std, alpha=0.5)

    def plot_data(self, line=True, color='tab:gray'):
        if line:
            plt.plot(self.x_arr.arr, self.t_arr.arr, ':.', color=color, alpha=0.5)
        else:
            plt.plot(self.x_arr.arr, self.t_arr.arr, '.', color=color)
            
    
    def plot_f_gp(self, gp, std=None, std_flag=True, color=None):
        mu, sigma = self.cal_pred_mu_sigma(gp)

        if torch.is_tensor(mu) and mu.requires_grad:
            mu = mu.detach()
            sigma = sigma.detach()

        diagonal = get_mat_op('diagonal', dtype=self.x_arr.dtype)
        sqrt = get_mat_op('sqrt', dtype=self.x_arr.dtype)

        if std is None:
            std = sqrt(diagonal(sigma))
        x_arr = gp.x_arr.arr

        x_arr = x_arr.flatten()
        mu = mu.flatten()

        if color:
            plt.plot(x_arr, mu, '--', color=color)
        else:
            plt.plot(x_arr, mu, '--')

        if std_flag:
            if color:
                plt.fill_between(x_arr, mu - std, mu + std, color=color, alpha=0.5)
            else:
                plt.fill_between(x_arr, mu - std, mu + std, alpha=0.5)


class GPMixture:
    def __init__(self, x_list: List[Array] = None,
                 t_list: List[Array] = None,
                 n_max_comp=None,
                 n_min_comp=1,
                 min_ws=0.05,
                 lengthscale_bounds=(0.5, 1.5),
                 sigma_f_bounds=(0.1, 1.5),
                 sigma_n_bounds=(0.1, 1),
                 fixed_inference_flag=False,
                 learning_rate=1e-3,
                 max_epoch=10,
                 n_trial=5,
                 nround=20,
                 jitter=1e-10,
                 debug=False):
        # for simulation purpose
        if x_list is None and t_list is None:
            return

        assert len(x_list) == len(t_list)
        for x, t in zip(x_list, t_list):
            assert type(x) is Array
            assert type(t) is Array
            assert len(x) == len(t)
            assert x.arr.shape == t.arr.shape

        self.x_list = x_list
        self.t_list = t_list
        self.n_data_item = len(x_list)

        self.n_max_comp = n_max_comp
        self.n_min_comp = n_min_comp

        self.min_ws = min_ws

        self.lengthscale_bounds = (torch.tensor(lengthscale_bounds[0]), torch.tensor(lengthscale_bounds[1]))
        self.sigma_f_bounds = sigma_f_bounds
        self.sigma_n_bounds = sigma_n_bounds
        self.sigma_f2_bounds = (torch.tensor(sigma_f_bounds[0] ** 2), torch.tensor(sigma_f_bounds[1] ** 2))
        self.sigma_n2_bounds = (torch.tensor(sigma_n_bounds[0] ** 2), torch.tensor(sigma_n_bounds[1] ** 2))

        self.fixed_inference_flag = fixed_inference_flag

        self.debug = debug

        self.n_trial = n_trial
        self.nround = nround
        self.jitter = jitter

        # torch related parameters
        self.torch_dtype = torch.float64
        self.optimizer = None
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.pos_infinite = float("inf")
        self.neg_infinite = float('-inf')

        # component parameters
        self.xf_arr = None
        self.para = None  # Parameters, container holding l, sigma_f2, sigma_n2, pi
        self.data_gp_list = None  # List[GaussianProcess], for each data point
        self.data_norm_para = Parameters()

    def clamp(self, para: Tensor, para_range: List[Tensor]):
        """
        clamp a parameter to a given range
        :param para: a parameter
        :param para_range: range specified for this parameter
        :return: clamped value
        """
        if para < para_range[0]:
            para.data.copy_(para_range[0].data)
        elif para > para_range[1]:
            para.data.copy_(para_range[1].data)

    # sample a number from a given range
    def rnd_unif(self, para_range: List[Tensor], n_number=1):
        return para_range[0] + (para_range[1] - para_range[0]) * torch.rand(n_number, dtype=self.torch_dtype)

    # preprocess raw data, perform standardization
    # each element of t_list must be n x 1
    def preproc_data(self, x_list, t_list):
        # convert data into tensor, reshape
        self.data_gp_list = list()
        for x, t in zip(x_list, t_list):
            if x.dtype == 'numpy':
                x = x.to_tensor()
            if t.dtype == 'numpy':
                t = t.to_tensor()
            x.arr = x.arr.flatten()
            # print(f'before={x.arr}')
            x.arr = x.arr / max(x.arr)  # so all x are in [0, 1]
            # print(f'after={x.arr}\n')
            t.arr = t.arr.reshape((-1, 1))
            self.data_gp_list.append(GaussianProcess(x_arr=x, t_arr=t))

        # set xf_arr and normalize data
        tmpgp = GaussianProcess()
        tmpgp = sum([gp for gp in self.data_gp_list], start=tmpgp)

        min_val, max_val = tmpgp.x_arr.arr.min(), tmpgp.x_arr.arr.max()
        if self.xf_arr is None:
            self.xf_arr = Array(torch.linspace(min_val, max_val, steps=50 + 1, dtype=self.torch_dtype), dtype='torch')

        xmu, xstd = float(tmpgp.x_arr.arr.mean()), float(tmpgp.x_arr.arr.std())
        tmu, tstd = float(tmpgp.t_arr.arr.mean()), float(tmpgp.t_arr.arr.std())
        self.data_norm_para.xmu = xmu
        self.data_norm_para.xstd = xstd
        self.data_norm_para.tmu = tmu
        self.data_norm_para.tstd = tstd
        self.norm_data(xmu, xstd, tmu, tstd)

    def norm_data(self, xmu, xstd, tmu, tstd):
        if self.data_gp_list is None:
            return
        self.xf_arr.norm(mu=xmu, std=xstd)
        for gp in self.data_gp_list:
            gp.x_arr.norm(mu=xmu, std=xstd)
            gp.t_arr.norm(mu=tmu, std=tstd)
            if gp.y_arr:
                gp.y_arr = gp.y_arr.norm(mu=tmu, std=tstd)

    def init_para(self, n_component):
        para = Parameters(dtype='torch')
        lengthscale = self.rnd_unif(self.lengthscale_bounds)
        f2 = self.rnd_unif([lengthscale * b for b in self.sigma_f_bounds]) ** 2
        n2 = self.rnd_unif(self.sigma_n_bounds) ** 2
        para.l = lengthscale.clone().detach().requires_grad_(True)
        para.sigma_f2 = f2.clone().detach().requires_grad_(True)
        para.sigma_n2 = n2.clone().detach().requires_grad_(True)

        para.N = self.n_data_item
        para.K = n_component
        para.early_break_flag = False  # early break in case a component's weight is less than min_ws
        para.ws = torch.rand(n_component, dtype=self.torch_dtype) + 1
        para.ws = para.ws / sum(para.ws)

        comp_ker_para = Parameters(dtype='torch')
        comp_ker_para.sigma_f2 = para.sigma_f2
        comp_ker_para.sigma_n2 = self.jitter
        comp_ker_para.l = para.l

        dataitem_ker_para = Parameters(dtype='torch')
        dataitem_ker_para.sigma_f2 = para.sigma_f2
        dataitem_ker_para.sigma_n2 = para.sigma_n2
        dataitem_ker_para.l = para.l

        para.comp_ker_para = comp_ker_para
        para.dataitem_ker_para = dataitem_ker_para

        # initialize the components
        u_mat = torch.randn((n_component, len(self.xf_arr)), dtype=self.torch_dtype)
        para.components_gp_list = list()
        for k in range(n_component):
            tmpgp = GaussianProcess(x_arr=self.xf_arr, kernel_para=comp_ker_para, inv_flag=True)
            cholesky = get_mat_op('cholesky', dtype='torch')
            matmul = get_mat_op('matmul', dtype='torch')
            tmp_L_mat = cholesky(tmpgp.covmat)  # Tensor
            tmpgp.t_arr = Array(matmul(tmp_L_mat, u_mat[k].reshape(-1, 1)).detach(), self.t_list[0].mu,
                                self.t_list[0].std,
                                dtype=self.t_list[0].dtype)
            tmpgp.y_arr = tmpgp.t_arr
            para.components_gp_list.append(tmpgp)

        # update gps for data points
        for gp in self.data_gp_list:
            gp.update_kernel_para(dataitem_ker_para)

        # configure optimizer
        self.optimizer = torch.optim.Adam([para.l, para.sigma_f2, para.sigma_n2], lr=self.learning_rate)
        return para

    def em_optim0(self, n_component):
        n_trial = self.n_trial
        lb_arr = torch.zeros(n_trial, dtype=self.torch_dtype) + self.neg_infinite
        bic_arr = torch.zeros(n_trial, dtype=self.torch_dtype) + self.pos_infinite
        res_list = list()

        for i in range(n_trial):
            if self.debug:
                print('-' * 15, f'K={n_component} | i_trial={i + 1} | n_trial={n_trial}', '-' * 15)

            # para = self.init_para(n_component)
            # para = self.em_algo(para)
            # res_list.append(para)
            # lb_arr[i] = res_list[i].lb_arr[-1]
            # bic_arr[i] = res_list[i].bic

            try:
                para = self.init_para(n_component)
                para = self.em_algo(para)
                res_list.append(para)
                lb_arr[i] = res_list[i].lb_arr[-1]
                bic_arr[i] = res_list[i].bic
            except Exception:
                print(f'Exception at i_trial={i}')

            if self.debug:
                para.disp(title='estimated parameters')

        min_ind = torch.argmin(bic_arr)
        res = res_list[min_ind]

        res.disp(title=f'Estimated Parameters. k={n_component}')

        return res

    # perform inference for K components
    def em_algo(self, para, fixed_inference_flag=False):
        lb = self.neg_infinite
        lb_arr = []

        log_zmat = self.update_log_zmat(para)  # p(t_n | y_k, x_n, xf, theta )

        for i in range(self.nround):
            if self.debug:
                print('iteration=', i + 1, '  lb=', lb)

            # E-Step
            Z = self.norm_log_zmat(log_zmat)

            # M-Step
            # para = self.mstep(para, Z)
            if fixed_inference_flag:
                para = self.mstep_fixed(para, Z)
            else:
                para = self.mstep(para, Z)

            if para.early_break_flag:
                if self.debug:
                    print('early break')
                lb_new = self.elbo(Z, log_zmat)
                lb_arr.append(lb_new)
                break

            # update intermediate variables
            for gp in para.components_gp_list:
                gp.update_kernel_para(para.comp_ker_para)
            # for gp in self.data_gp_list:
            #     gp.update_kernel_para(para.dataitem_ker_para)

            log_zmat = self.update_log_zmat(para)
            lb_new = self.elbo(Z, log_zmat)
            lb_arr.append(lb_new)

            if abs(lb_new - lb) < abs(1e-6 * lb):
                break
            else:
                lb = lb_new

        if self.debug:
            if i == self.nround:
                print(f'Run all {i + 1} iterations. lb={lb}')
            else:
                print(f'Converge in  {i + 1} iterations. lb={lb}')

        bic = self.cal_bic(Z, log_zmat)
        print(f'bic={bic}')

        if self.debug:
            nd = len(lb_arr)
            if nd >= 3:
                plt.plot(list(range(nd - 3)), lb_arr[3:nd])
                plt.show()

        if not fixed_inference_flag:
            para.title = 'Estimated parameters'
        para.bic = bic
        para.lb_arr = lb_arr
        para.lb = lb_new

        return para

    # calculate log marginal likelihood
    def cal_loglik(self, Z, log_zmat):
        loglik = 0.0
        inds = Z > 1e-20  # skip low weight elements
        loglik = loglik + sum(Z[inds] * log_zmat[inds])
        return loglik

    def entropy(self, Z):
        inds = Z > 1e-20
        return sum(Z[inds] * torch.log(Z[inds]))

    def elbo(self, Z, log_zmat):
        res = self.cal_loglik(Z, log_zmat) - self.entropy(Z)
        return res.detach_()

    def cal_bic(self, Z, log_zmat):
        N, K = Z.shape
        # pi, sigma_f2, l, sigma_n2
        res = -2 * self.elbo(Z, log_zmat) + 4 * K * math.log(N)  # the smaller bic, the better model
        return res.detach_()

    # log_zmat is log likelihood
    @staticmethod
    def norm_log_zmat(log_zmat):
        Z = torch.exp(log_zmat - torch.max(log_zmat, axis=1, keepdims=True).values)
        Z = Z / torch.sum(Z, axis=1, keepdims=True)
        return Z.detach_()

    @staticmethod
    def norm_z(Z):
        Z = Z / torch.sum(Z, axis=1, keepdims=True)
        return Z.detach_()

    def update_log_zmat(self, para):
        log_zmat = torch.zeros((para.N, para.K), dtype=self.torch_dtype)
        for n in range(para.N):
            for k in range(para.K):
                # log_zmat[n, k] = torch.log(para.ws[k]) + para.components_gp_list[k].cal_pred_loglik(self.data_gp_list[n])
                log_zmat[n, k] = torch.log(para.ws[k]) + \
                                 para.components_gp_list[k].approx_pred_loglik(self.data_gp_list[n])
        return log_zmat

    # maximize ws given Z
    def maximize_ws(self, Z):
        ws = torch.sum(Z, axis=0) / Z.shape[0]
        return ws.detach_()

    # maximize hidden functions given Z
    def maximize_component_mu(self, para, Z):
        label_arr = self.get_label(Z)
        comps_gp_list = []

        f_ker_para = Parameters()
        f_ker_para.l = para.l
        f_ker_para.sigma_f2 = para.sigma_f2
        f_ker_para.sigma_n2 = para.sigma_n2.detach()

        for k in range(para.K):
            inds = [i for i, label in enumerate(label_arr) if label == k]
            if len(inds) == 0:
                # para.components_gp_list[k].t_arr.arr.detach_()
                continue

            tmpgp = GaussianProcess()
            tmpgp = sum([self.data_gp_list[i] for i in inds], start=tmpgp)
            x_arr = tmpgp.x_arr.arr.numpy().flatten()  # 1 x n
            t_arr = tmpgp.t_arr.arr.numpy()  # n x 1
            idx = x_arr.argsort()
            x_arr = x_arr[idx]
            t_arr = t_arr[idx]

            n_total_items = 50
            n_bin = 10
            n_items_per_bin = round(n_total_items / n_bin)
            boarder = np.linspace(x_arr[0], x_arr[-1], num=n_bin + 1, endpoint=True)
            en_arr = boarder[1:]

            sample_inds = []
            st = 0
            for i in range(n_bin):
                pos = bisect.bisect(x_arr, en_arr[i], lo=st)
                sample_inds += sorted(random.sample(list(range(st, pos)), min(n_items_per_bin, pos - st)))
                st = pos

            x_arr = x_arr[sample_inds]
            t_arr = t_arr[sample_inds]
            tmpgp.x_arr = Array(torch.from_numpy(x_arr), dtype='torch')
            tmpgp.t_arr = Array(torch.from_numpy(t_arr), dtype='torch')

            tmpgp.inv_flag = True
            comps_gp_list.append(tmpgp)

        curr_loglik = self.pos_infinite
        # minus_loglik = 0.0

        for i in range(self.max_epoch):
            self.optimizer.zero_grad()
            tmp_res_list = []
            for tmpgp in comps_gp_list:
                tmpgp.update_kernel_para(f_ker_para)
                tmp_res_list.append(tmpgp.cal_marginal_loglik())
                # minus_loglik = minus_loglik - tmpgp.cal_marginal_loglik()  # will cause in place calculation problem
            if self.debug:
                print(f'i={i} -loglik={curr_loglik}')

            minus_loglik = sum(tmp_res_list)
            minus_loglik.backward(retain_graph=True)
            self.optimizer.step()
            if abs((minus_loglik.item() - curr_loglik) / curr_loglik) < 1e-6:
                break
            curr_loglik = minus_loglik.item()
            self.clamp(f_ker_para.l, self.lengthscale_bounds)
            self.clamp(f_ker_para.sigma_f2, self.sigma_f2_bounds)

        for k, tmpgp in enumerate(comps_gp_list):
            mu = tmpgp.cal_pred_mu(para.components_gp_list[k]).detach()
            para.components_gp_list[k].t_arr.arr = 0.5 * (mu.detach_() + para.components_gp_list[k].t_arr.arr.detach())
            para.components_gp_list[k].y_arr = para.components_gp_list[k].t_arr

        # update the kernel parameters of components to facilitate pytorch optimization
        for gp in para.components_gp_list:
            gp.update_kernel_para(gp.kernel_para)

    # maximize hidden functions given Z and fixed parameter
    def maximize_component_mu_fixed(self, para, Z):
        label_arr = self.get_label(Z)

        f_ker_para = Parameters()
        f_ker_para.l = para.l
        f_ker_para.sigma_f2 = para.sigma_f2
        f_ker_para.sigma_n2 = para.sigma_n2.detach()

        for k in range(para.K):
            inds = [i for i, label in enumerate(label_arr) if label == k]
            if len(inds) == 0:
                # para.components_gp_list[k].t_arr.arr.detach_()
                continue

            tmpgp = GaussianProcess()
            tmpgp = sum([self.data_gp_list[i] for i in inds], start=tmpgp)
            x_arr = tmpgp.x_arr.arr.numpy().flatten()  # 1 x n
            t_arr = tmpgp.t_arr.arr.numpy()  # n x 1
            idx = x_arr.argsort()
            x_arr = x_arr[idx]
            t_arr = t_arr[idx]

            n_total_items = 50
            n_bin = 10
            n_items_per_bin = round(n_total_items / n_bin)
            boarder = np.linspace(x_arr[0], x_arr[-1], num=n_bin + 1, endpoint=True)
            en_arr = boarder[1:]

            sample_inds = []
            st = 0
            for i in range(n_bin):
                pos = bisect.bisect(x_arr, en_arr[i], lo=st)
                sample_inds += sorted(random.sample(list(range(st, pos)), min(n_items_per_bin, pos - st)))
                st = pos

            x_arr = x_arr[sample_inds]
            t_arr = t_arr[sample_inds]
            tmpgp.x_arr = Array(torch.from_numpy(x_arr), dtype='torch')
            tmpgp.t_arr = Array(torch.from_numpy(t_arr), dtype='torch')

            tmpgp.inv_flag = True
            tmpgp.update_kernel_para(f_ker_para)

            mu = tmpgp.cal_pred_mu(para.components_gp_list[k]).detach()
            para.components_gp_list[k].t_arr.arr = mu
            para.components_gp_list[k].y_arr = para.components_gp_list[k].t_arr

    def maximize_theta(self, para, Z):
        curr_loglik = self.pos_infinite
        for i in range(self.max_epoch):
            # self.update_para(para)
            self.optimizer.zero_grad()
            log_zmat = self.update_log_zmat(para)
            minus_loglik = -self.cal_loglik(Z, log_zmat)
            if self.debug:
                print(f'i={i} -loglik={curr_loglik}')
            minus_loglik.backward(retain_graph=True)
            self.optimizer.step()
            if abs((minus_loglik.item() - curr_loglik) / curr_loglik) < 1e-6:
                break
            curr_loglik = minus_loglik.item()

            self.clamp(para.sigma_n2, self.sigma_n2_bounds)

        return -minus_loglik.item()

    def mstep_fixed(self, para, Z):
        para.ws = self.maximize_ws(Z)
        self.maximize_component_mu_fixed(para, Z)
        return para

    def mstep(self, para, Z):
        # avoid division by zero
        if any(sum(Z) < 1e-8):
            para.early_break_flag = True
            Z[:, sum(Z) < 1e-8] = 1e-8
            Z = self.norm_z(Z)

        para.ws = self.maximize_ws(Z)
        if any(para.ws < self.min_ws / 2):
            para.early_break_flag = True

        # optimize l, sigma_f2
        self.maximize_component_mu(para, Z)

        # optimize sigma_n2
        loglik = self.maximize_theta(para, Z)
        para.loglik = loglik

        return para

    # remove components with weight less than min_ws
    def rm_component(self, para):
        rm_inds = [i for i in range(para.K) if para.ws[i] < self.min_ws]
        if len(rm_inds) == 0:
            return para

        print(f'Remove components {rm_inds} with weight less than min_ws={self.min_ws}.')
        keep_inds = [i for i in range(para.K) if not para.ws[i] < self.min_ws]

        para.K = len(keep_inds)
        para.ws = para.ws[keep_inds] / sum(para.ws[keep_inds])
        para.components_gp_list = [para.components_gp_list[i] for i in keep_inds]
        para = self.em_algo(para, fixed_inference_flag=True)

        return para

    def get_label(self, Z):
        label_arr = torch.argmax(Z, axis=1)
        return label_arr.detach()

    def fixed_infer(self, para):
        back_data_gp_list = self.data_gp_list

        self.data_gp_list = para.data_gp_list
        log_zmat = self.update_log_zmat(para)  # p(t_n | y_k, x_n, xf, theta )
        Z = self.norm_log_zmat(log_zmat)
        lb = self.elbo(Z, log_zmat)
        bic = self.cal_bic(Z, log_zmat)
        para.bic = bic
        para.lb = lb

        self.data_gp_list = back_data_gp_list

        return para

    def run(self):
        n_run = self.n_max_comp - self.n_min_comp + 1
        bic_arr = torch.zeros(n_run, dtype=self.torch_dtype) + self.pos_infinite  # np.full(n_run, self.pos_infinite)
        res_list = list()

        # preprocessing data
        self.preproc_data(self.x_list, self.t_list)

        for i, n_comp in enumerate(range(self.n_max_comp, self.n_min_comp - 1, -1)):
            print()
            print(20 * '*' + ' k = ' + str(n_comp) + ' ' + 20 * '*')
            res = self.em_optim0(n_comp)
            res_list.append(res)
            bic_arr[i] = res.bic

        min_ind = torch.argmin(bic_arr)
        res = res_list[min_ind]

        print('\n', '*' * 20, 'Finish.', '*' * 20, '\n')
        res.disp(title=f'Best result, K={res.K}')

        res = self.rm_component(res)
        log_zmat = self.update_log_zmat(res)
        Z = self.norm_log_zmat(log_zmat)
        res.label_arr = self.get_label(Z)

        print()
        res.disp(title=f'Final Result, K={res.K}')

        self.para = res
        return res

    def plot(self, title='', label_arr=None, line=True, n_max_line_per_comp=20):
        if label_arr is None:
            for gp in self.data_gp_list:
                gp.plot_data(line=line)
            for k, gp in enumerate(self.para.components_gp_list):
                gp.plot_f(np.sqrt(self.para.sigma_n2.detach().numpy()), color=COLORS[k])
            return

        for k, gp in enumerate(self.para.components_gp_list):
            inds = [i for i, label in enumerate(label_arr) if label == k]
            if len(inds) > n_max_line_per_comp:
                inds = random.sample(inds, n_max_line_per_comp)
            for i in inds:
                self.data_gp_list[i].plot_data(line=line, color=COLORS[k])
            gp.plot_f(np.sqrt(self.para.sigma_n2.detach().numpy()), color=COLORS[k])

        plt.title(title)
        plt.show()

    # simulate data
    def simulate(self, para, dtype='torch'):
        """
        :param para: N, K, l, sigma_f2, sigma_n2
        :return: x_list, y_list, t_list, label_arr
        """
        if hasattr(para, 'seed') and para.seed:
            np.random.seed(para.seed)

        para.dtype = dtype
        label_arr = np.zeros(para.N, dtype=int)
        ws = np.random.rand(para.K) + 1
        ws = ws / sum(ws)
        border = np.round(para.N * np.cumsum(ws)).astype(int)
        st_arr = [0] + border[0:-1].tolist()
        en_arr = border.tolist()
        for k in range(para.K):
            label_arr[st_arr[k]:en_arr[k]] = k

        if dtype == 'torch':
            para.ws = torch.tensor(ws)
            para.label_arr = torch.tensor(label_arr)
        else:
            para.ws = ws
            para.label_arr = label_arr

        para.jitter = 1e-10

        comp_ker_para = Parameters(dtype=dtype)
        comp_ker_para.sigma_f2 = para.sigma_f2
        comp_ker_para.sigma_n2 = para.jitter
        comp_ker_para.l = para.l

        dataitem_ker_para = Parameters(dtype=dtype)
        dataitem_ker_para.sigma_f2 = para.sigma_f2
        dataitem_ker_para.sigma_n2 = para.sigma_n2
        dataitem_ker_para.l = para.l

        para.comp_ker_para = comp_ker_para

        # initialize the components
        xf_arr = Array(np.arange(0.0, 4.1, 0.1))
        if dtype == 'torch':
            xf_arr = xf_arr.to_tensor()
        para.xf_arr = xf_arr
        para.components_gp_list = list()
        para.u_mat = np.random.randn(para.K, len(xf_arr))
        if dtype == 'torch':
            para.u_mat = torch.tensor(para.u_mat)
        for k in range(para.K):
            tmpgp = GaussianProcess(x_arr=xf_arr, kernel_para=comp_ker_para, inv_flag=True)
            cholesky = get_mat_op('cholesky', dtype=dtype)
            matmul = get_mat_op('matmul', dtype=dtype)
            tmp_L_mat = cholesky(tmpgp.covmat)  # Tensor
            tmpgp.t_arr = Array(matmul(tmp_L_mat, para.u_mat[k].reshape(-1, 1)), dtype=dtype)
            tmpgp.y_arr = tmpgp.t_arr
            para.components_gp_list.append(tmpgp)

        data_gp_list = list()
        x_list = list()
        for i in range(para.N):
            tmpn = int(np.round(np.random.rand(1) * 4 + 3))
            tmpx = Array(np.sort(np.random.rand(tmpn)).reshape(-1, 1) * 4)
            if dtype == 'torch':
                tmpx = tmpx.to_tensor()
            x_list.append(tmpx)
            data_gp_list.append(GaussianProcess(x_arr=tmpx))

        y_list = list()
        t_list = list()
        para.comp_sigma_n2_arr = (0.5 + np.random.rand(para.K)) ** 2 * para.sigma_n2
        for n, k in enumerate(label_arr):
            comp_gp = para.components_gp_list[k]
            mu = comp_gp.cal_pred_mu(data_gp_list[n])
            noise = np.sqrt(para.comp_sigma_n2_arr[k]) * np.random.randn(len(mu)).reshape(-1, 1)
            # noise = np.sqrt(para.sigma_n2) * np.random.randn(len(mu)).reshape(-1, 1)
            if dtype == 'torch':
                noise = torch.tensor(noise)
            data_gp_list[n].y_arr = Array(mu, dtype=dtype)
            data_gp_list[n].t_arr = Array(mu + noise, dtype=dtype)
            y_list.append(Array(mu, dtype=dtype))
            t_list.append(Array(mu + noise, dtype=dtype))

            data_gp_list[n].update_kernel_para(dataitem_ker_para)

        para.data_gp_list = data_gp_list
        para.x_list = x_list
        para.y_list = y_list
        para.t_list = t_list

        para.title = 'simulation parameters'
        para.disp(title=para.title)

        if para.plot_flag:
            for k, gp in enumerate(para.components_gp_list):
                inds = [i for i, label in enumerate(label_arr) if label == k]
                if len(inds) > 30:
                    inds = random.sample(inds, 30)
                for i in inds:
                    para.data_gp_list[i].plot_data(line=True, color=COLORS[k])
                gp.plot_f(np.sqrt(para.sigma_n2), color=COLORS[k])
            # for gp in para.components_gp_list:
            #     gp.plot_f(np.sqrt(para.sigma_n2))
            # for gp in para.data_gp_list:
            #     gp.plot_data()
            plt.title('Ground truth')
            plt.show()

        return para

"""
def gen_data(seed=12345):
    real_para = Parameters()
    real_para.K = 3
    real_para.N = real_para.K * 30
    real_para.l = 1.0
    real_para.sigma_f2 = 1 ** 2
    real_para.sigma_n2 = 0.3 ** 2
    real_para.seed = seed
    real_para.plot_flag = True

    tmpgpm = GPMixture()
    real_para = tmpgpm.simulate(real_para, dtype='torch')
    x_list, t_list, xf_arr = real_para.x_list, real_para.t_list, real_para.xf_arr
    # x_list = [x.to_tensor() for x in real_para.x_list]
    # t_list = [t.to_tensor() for t in real_para.t_list]
    # xf_arr = real_para.xf_arr.to_tensor()

    return real_para, x_list, t_list, xf_arr


import pickle

if __name__ == '__main__':
    real_para, x_list, t_list, xf_arr = gen_data(seed=12345)

    gpm = GPMixture(x_list=x_list, t_list=t_list, n_max_comp=5, n_min_comp=2, n_trial=10,
                    learning_rate=0.001,
                    debug=False)

    gpm.preproc_data(x_list, t_list)
    real_para.l = real_para.l / gpm.data_norm_para.xstd
    real_para.sigma_f2 = real_para.sigma_f2 / gpm.data_norm_para.tstd ** 2
    real_para.sigma_n2 = real_para.sigma_n2 / gpm.data_norm_para.tstd ** 2
    real_para = gpm.fixed_infer(real_para)
    real_para.disp(title='Estimation with ground truth')

    filename = 'res.pkl'
    load_saved_res_flag = False

    import time

    start_time = time.time()
    if load_saved_res_flag:
        with open(filename, 'rb') as fh:
            res, gpm = pickle.load(fh)
    else:
        res = gpm.run()

    gpm.plot(title='Final estimation', label_arr=res.label_arr)

    print()
    print("--- %s seconds ---" % (time.time() - start_time))

    if not load_saved_res_flag:
        with open(filename, 'wb') as fh:
            pickle.dump([res, gpm], fh)
            
            
            """