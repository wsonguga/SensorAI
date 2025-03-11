import numpy as np
import seaborn as sns
import sys
import time, datetime
import pytz
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt

from numba import jit
import numpy as np
from numpy.linalg import norm

def start_SST(startdata,win_length,n_component,order,lag):
    Score_start=np.zeros(1) # get the initial score, Score_start
    x1 = np.empty(order, dtype=np.float64) 
    x1 = np.random.rand(order)
    x1 /= np.linalg.norm(x1)
    score_start, x = SingularSpectrumTransformation(win_length=win_length, x0=x1, n_components=n_component,order=order, lag=lag,is_scaled=True).score_online(startdata)
    Score_start=score_start+Score_start*10**7
    return Score_start,x


#### After that we could use the output of start_SST function to initialize the infinite loop

def stream_SST(stream,win_length,n_component,order,lag,x0):  #state_last,thres1,thres2
  ### stream is the new data coming through
  ### last data is the data from the last second
  starttime=time.time()
  
  #data=np.concatenate((lastdata[lag:], stream), axis=None)
  data=stream
  score, x1 = SingularSpectrumTransformation(win_length=win_length, x0=x0, n_components=n_component,order=order, lag=lag,is_scaled=True).score_online(data)
  score=score*10**5

  end=time.time()
  duration=end-starttime
  return score,duration,x1



##### To be noticed ! state and x1 would be used as input for stream_SST function in each loop 


class SingularSpectrumTransformation():
    """SingularSpectrumTransformation class."""

    def __init__(self, win_length, x0, n_components=2, order=None, lag=None,
                 is_scaled=False, use_lanczos=True, rank_lanczos=None, eps=1e-3):
        """Change point detection with Singular Spectrum Transformation.
        Parameters
        ----------
        win_length : int
            window length of Hankel matrix.
        n_components : int
            specify how many rank of Hankel matrix will be taken.
        order : int
            number of columns of Hankel matrix.
        lag : int
            interval between history Hankel matrix and test Hankel matrix.
        is_scaled : bool
            if false, min-max scaling will be applied(recommended).
        use_lanczos : boolean
            if true, Lanczos method will be used, which makes faster.
        rank_lanczos : int
            the rank which will be used for lanczos method.
            for the detail of lanczos method, see [1].
        eps : float
            specify how much noise will be added to initial vector for
            power method.
            (FELIX: FEedback impLIcit kernel approXimation method)
            for the detail, see [2].
        References
        ----------
        [1]: Tsuyoshi Ide et al., Change-Point Detection using Krylov Subspace Learning
        [2]: Tsuyoshi Ide, Speeding up Change-Point Detection using Matrix Compression (Japanse)
        """
        self.win_length = win_length
        self.n_components = n_components
        self.order = order
        self.lag = lag
        self.is_scaled = is_scaled
        self.use_lanczos = use_lanczos
        self.rank_lanczos = rank_lanczos
        self.eps = eps
        self.x0=x0

    def score_online(self, x):
        """Calculate anomaly score (offline).
        Parameters
        ----------
        x : 1d numpy array
            input time series data.
        Returns
        -------
        score : 1d array
            change point score.
        """
        if self.order is None:
            # rule of thumb
            self.order = self.win_length
        if self.lag is None:
            # rule of thumb
            self.lag = self.order // 2
        if self.rank_lanczos is None:
            # rule of thumb
            if self.n_components % 2 == 0:
                self.rank_lanczos = 2 * self.n_components
            else:
                self.rank_lanczos = 2 * self.n_components - 1

        assert isinstance(x, np.ndarray), "input array must be numpy array."
        assert x.ndim == 1, "input array dimension must be 1."
        assert isinstance(self.win_length, int), "window length must be int."
        assert isinstance(self.n_components, int), "number of components must be int."
        assert isinstance(self.order, int), "order of partial time series must be int."
        assert isinstance(self.lag, int), "lag between test series and history series must be int."
        assert isinstance(self.rank_lanczos, int), "rank for lanczos must be int."
        # assert self.win_length + self.order + self.lag < x.size, "data length is too short."

        # all values should be positive for numerical stabilization
        # if not self.is_scaled:
        #     x_scaled = MinMaxScaler(feature_range=(1, 2))\
        #         .fit_transform(x.reshape(-1, 1))[:, 0]
        # else:
        x_hist = x[:self.win_length+self.lag]
        x_new = x[self.lag:]
        score, x1 = _score_online(x_hist, x_new, self.x0, self.order,
            self.win_length, self.lag, self.n_components, self.rank_lanczos,
            self.eps, use_lanczos=self.use_lanczos)

        return score, x1

@jit(nopython=True)
def _score_online(x, y, x0, order, win_length, lag, n_components, rank, eps, use_lanczos):
    """Core implementation of offline score calculation."""
    # start_idx = win_length + order + lag + 1
    # end_idx = x.size + 1


    score = np.zeros(1)
    # for t in range(start_idx, end_idx):
    # compute score at each index

    # get Hankel matrix
    X_history = _create_hankel(x, order,
        start=order,
        end=win_length)

    X_test = _create_hankel(y, order,
        start=order,
        end=win_length)


    if use_lanczos:
        score, x1 = _sst_lanczos(X_test, X_history, n_components,
                                      rank, x0)
        # update initial vector for power method
        x0 = x1 + eps * np.random.rand(x0.size)
        x0 /= np.linalg.norm(x0)
    else:
        score = _sst_svd(X_test, X_history, n_components)

    return score,x0


@jit(nopython=True)
def _create_hankel(x, order, start, end):
    """Create Hankel matrix.
    Parameters
    ----------
    x : full time series
    order : order of Hankel matrix
    start : start index
    end : end index
    Returns
    -------
    2d array shape (window length, order)
    """
    win_length = end - start
    X = np.empty((win_length, order))
    for i in range(order):
        X[:, i] = x[(start - i):(end - i)]
    return X


@jit(nopython=True)
def _sst_lanczos(X_test, X_history, n_components, rank, x0):
    """Run sst algorithm with lanczos method (FELIX-SST algorithm)."""
    P_history = X_history.T @ X_history
    P_test = X_test.T @ X_test
    # calculate the first singular vec of test matrix
    u, _, _ = power_method(P_test, x0, n_iter=1)
    T = lanczos(P_history, u, rank)
    vec, val = eig_tridiag(T)
    return 1 - (vec[0, :n_components] ** 2).sum(), u


@jit("f8(f8[:,:],f8[:,:],u1)", nopython=True)
def _sst_svd(X_test, X_history, n_components):
    """Run sst algorithm with svd."""
    U_test, _, _ = np.linalg.svd(X_test, full_matrices=False)
    U_history, _, _ = np.linalg.svd(X_history, full_matrices=False)
    _, s, _ = np.linalg.svd(U_test[:, :n_components].T @
        U_history[:, :n_components], full_matrices=False)
    return 1 - s[0]



@jit(nopython=True)
def power_method(A, x0, n_iter=1):
    """Compute the first singular components by power method."""
    for i in range(n_iter):
        x0 = A.T @ A @ x0

    v = x0 / norm(x0)
    s = norm(A @ v)
    u = A @ v / s

    return u, s, v


@jit(nopython=True)
def lanczos(C, a, s):
    """Perform lanczos algorithm."""
    # initialization
    r = np.copy(a)
    a_pre = np.zeros_like(a, dtype=np.float64)
    beta_pre = 1
    T = np.zeros((s, s))

    for j in range(s):
        a_post = r / beta_pre
        alpha = a_post.T @ C @ a_post
        r = C @ a_post - alpha*a_post - beta_pre*a_pre
        beta_post = norm(r)

        T[j, j] = alpha
        if j - 1 >= 0:
            T[j, j-1] = beta_pre
            T[j-1, j] = beta_pre

        # update
        a_pre = a_post
        beta_pre = beta_post

    return T


@jit(nopython=True)
def eig_tridiag(T):
    """Compute eigen value decomposition for tridiag matrix."""
    # TODO: efficient implementation
    # ------------------------------------------------------
    # Is it really necessary to implement fast eig computation ?
    # The size of matrix T is practically at most 20 since almost 2 times
    # larger than n_components. Therefore fast implementation such as
    # QL algorithm may not provide computational cost benefit in total.
    # ------------------------------------------------------
    u, s, _ = np.linalg.svd(T)
    # NOTE: return value must be ordered
    return u, s