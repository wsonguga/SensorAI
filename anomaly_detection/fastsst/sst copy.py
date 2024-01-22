# -*- coding: utf-8 -*-
"""Singluar Spectrum Transformation.

The MIT License (MIT)
Copyright (c) 2018 statefb.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
import time
from numba import jit
from sklearn.preprocessing import MinMaxScaler
from .util.linear_algebra import power_method, lanczos, eig_tridiag


def start_SST(startdata,win_length,n_component,order,lag):
    if order == None: # IF added by sjc
        order = win_length
    Score_start=np.zeros(1) # get the initial score, Score_start
    x1 = np.empty(order, dtype=np.float64) 
    x1 = np.random.rand(order)
    x1 /= np.linalg.norm(x1)
    print("!!!!!!!!!!!!!!!!!!! startdata shape is ",startdata.shape)
    #score_start, x = SingularSpectrumTransformation(win_length=win_length, x0=x1, n_components=n_component,order=order, lag=lag,is_scaled=True).score_online(startdata)
    score_start, x = SingularSpectrumTransformation(win_length=win_length, n_components=n_component,order=order, lag=lag,is_scaled=True).score_offline(startdata)
    Score_start=score_start+Score_start*10**7
    return Score_start,x


#### After that we could use the output of start_SST function to initialize the infinite loop

def stream_SST(stream,win_length,n_component,order,lag):#,x0):  #state_last,thres1,thres2
  ### stream is the new data coming through
  ### last data is the data from the last second
  starttime=time.time()
  print("$$$$$$$$$$$$$$$$$$$$$$$ I'm in stream_SST")
  #data=np.concatenate((lastdata[lag:], stream), axis=None)
  data=stream
  #score, x1 = SingularSpectrumTransformation(win_length=win_length, x0=x0, n_components=n_component,order=order, lag=lag,is_scaled=True).score_online(data)
  score, x1 = SingularSpectrumTransformation(win_length=win_length, n_components=n_component,order=order, lag=lag,is_scaled=True).score_offline(data)
  score=score*10**5

  end=time.time()
  duration=end-starttime
  return score,duration,x1


class SingularSpectrumTransformation():
    """SingularSpectrumTransformation class."""

    def __init__(self, win_length, n_components=5, order=None, lag=None,
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
        if order == None:
            self.order = win_length
        else:
            self.order = order
        self.lag = lag
        self.is_scaled = is_scaled
        self.use_lanczos = use_lanczos
        self.rank_lanczos = rank_lanczos
        self.eps = eps

    def score_offline(self, x):
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
        print ("--------------------- x shape is ",x.shape)
        assert isinstance(x, np.ndarray), "input array must be numpy array."
        assert x.ndim == 1, "input array dimension must be 1."
        assert isinstance(self.win_length, int), "window length must be int."
        assert isinstance(self.n_components, int), "number of components must be int."
        assert isinstance(self.order, int), "order of partial time series must be int."
        assert isinstance(self.lag, int), "lag between test series and history series must be int."
        assert isinstance(self.rank_lanczos, int), "rank for lanczos must be int."
        #assert self.win_length + self.order + self.lag < x.size, "data length is too short."

        # all values should be positive for numerical stabilization
        if not self.is_scaled:
            x_scaled = MinMaxScaler(feature_range=(1, 2))\
                .fit_transform(x.reshape(-1, 1))[:, 0]
        else:
            x_scaled = x

        score = _score_offline(x_scaled, self.order,
            self.win_length, self.lag, self.n_components, self.rank_lanczos,
            self.eps, use_lanczos=self.use_lanczos)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&& score shape is ",score.shape)
        return score


@jit(nopython=True)
def _score_offline(x, order, win_length, lag, n_components, rank, eps, use_lanczos):
    """Core implementation of offline score calculation."""
    start_idx = win_length + order + lag + 1
    end_idx = x.size + 1

    # initialize vector for power method
    x0 = np.empty(order, dtype=np.float64)
    x0 = np.random.rand(order)
    x0 /= np.linalg.norm(x0)
    print("*************** i get to _score_offline")
    score = np.zeros_like(x)
    for t in range(start_idx, end_idx):
        # compute score at each index

        # get Hankel matrix
        X_history = _create_hankel(x, order,
            start=t - win_length - lag,
            end=t - lag)
        X_test = _create_hankel(x, order,
            start=t - win_length,
            end=t)

        if use_lanczos:
            score[t-1], x1 = _sst_lanczos(X_test, X_history, n_components,
                                          rank, x0)
            # update initial vector for power method
            x0 = x1 + eps * np.random.rand(x0.size)
            x0 /= np.linalg.norm(x0)
        else:
            score[t-1] = _sst_svd(X_test, X_history, n_components)
    print("*************** i get to return in _score_offline")
    return score


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
