import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from plotly.subplots import make_subplots
import chart_studio.plotly as py
import plotly.graph_objects as go

import re
import pytz
from datetime import datetime

import enum
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet, ElasticNetCV, MultiTaskElasticNet, MultiTaskElasticNetCV, GammaRegressor, HuberRegressor, Lars, LarsCV, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC, LinearRegression, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, PassiveAggressiveRegressor, Perceptron, PoissonRegressor, QuantileRegressor, RANSACRegressor, Ridge, RidgeCV, SGDRegressor, TheilSenRegressor, TweedieRegressor

from tslearn.neighbors import KNeighborsTimeSeriesRegressor
from tslearn.svm import TimeSeriesSVR

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, auc, roc_curve, roc_auc_score
from sklearn.metrics import PredictionErrorDisplay 

#import load_data as ld

algo_list = ['svr','nusvr','linear svr','ridge','ridge cv','linear regression','sgd','ard','bayesian ridge','passive aggressive','gamma','poisson','tweedie','huber','quantile','ranscar','thielsen','elasticnet','elasticnet cv','multitask elastic net','multitask elastic net cv','lars','lasso','lasso cv','lasso lars','lasso lars cv','lasso lars ic','orthogonal matching pursuit','orthogonal matching pursuit cv','ts knn','ts svr','perceptron']

# All inputs execpt random_state should be lists of values, even if only one value

# SUPPORT VECTOR MACHINE
def pipeBuild_SVR(kernel=['rbf'],degree=[3], gamma=['scale'],coef0=[0.0],tol=[1.0e-3],C=[1.0],
  shrinking=[True],cache_size=[200],verbose=[False],max_iter=[-1]):
  regressor = SVR()
  pipeline = Pipeline(steps=[('svr', regressor)])
  params = [{
        'svr__kernel': kernel,
        'svr__degree': degree,
        'svr__gamma': gamma,
        'svr__coef0': coef0,
        'svr__tol': tol,
        'svr__C': C,
        'svr__shrinking': shrinking,
        'svr__cache_size': cache_size,
        'svr__verbose': verbose,
        'svr__max_iter': max_iter,
    }]
  return pipeline, params

# NU-SUPPORT VECTOR MACHINE
def pipeBuild_NuSVR(nu=[0.5],kernel=['rbf'],degree=[3], gamma=['scale'],coef0=[0.0],tol=[1.0e-3],C=[1.0],
  shrinking=[True],cache_size=[200],verbose=[False],max_iter=[-1]):
  regressor = NuSVR()
  pipeline = Pipeline(steps=[('nusvr', regressor)])
  params = [{
        'nusvr__nu': nu,
        'nusvr__kernel': kernel,
        'nusvr__degree': degree,
        'nusvr__gamma': gamma,
        'nusvr__coef0': coef0,
        'nusvr__tol': tol,
        'nusvr__C': C,
        'nusvr__shrinking': shrinking,
        'nusvr__cache_size': cache_size,
        'nusvr__verbose': verbose,
        'nusvr__max_iter': max_iter,
    }]
  return pipeline, params

# LINEAR SUPPORT VECTOR MACHINE
def pipeBuild_LinearSVR(epsilon=[0.0],loss=['epsilon_insensitive'], fit_intercept=[True],
  intercept_scaling=[1.0],tol=[1.0e-4],C=[1.0],dual=[True],verbose=[False],max_iter=[1000],
  random_state=None):
  regressor = LinearSVR(random_state=random_state)
  pipeline = Pipeline(steps=[('lsvr', regressor)])
  params = [{
        'lsvr__epsilon': epsilon,
        'lsvr__loss': loss,
        'lsvr__fit_intercept': fit_intercept,
        'lsvr__intercept_scaling': intercept_scaling,
        'lsvr__tol': tol,
        'lsvr__C': C,
        'lsvr__dual': dual,
        'lsvr__verbose': verbose,
        'lsvr__max_iter': max_iter,
    }]
  return pipeline, params

# LINEAR REGRESSION
def pipeBuild_LinearRegression(fit_intercept=[True], copy_X=[True],n_jobs=[None],positive=[False],):
  regressor = LinearRegression()
  pipeline = Pipeline(steps=[('linreg', regressor)])
  params = [{
        'linreg__fit_intercept': fit_intercept,
        'linreg__copy_X': copy_X,
        'linreg__n_jobs': n_jobs,
        'linreg__positive': positive,
    }]
  return pipeline, params

# RIDGE
def pipeBuild_Ridge(alpha=[1.0],fit_intercept=[True], copy_X=[True],max_iter=[None],tol=[1.0e-4],):
  regressor = Ridge()
  pipeline = Pipeline(steps=[('ridge', regressor)])
  params = [{
        'ridge__alpha': alpha,
        'ridge__fit_intercept': fit_intercept,
        'ridge__copy_X': copy_X,
        'ridge__max_iter': max_iter,
        'ridge__tol': tol,
    }]
  return pipeline, params

# RIDGE CV
def pipeBuild_RidgeCV(alphas=[(0.1, 1.0, 10.0)],fit_intercept=[True], scoring=[None],cv=[None],
  gcv_mode=['auto'],store_cv_values=[False],alpha_per_target=[False]):
  regressor = RidgeCV()
  pipeline = Pipeline(steps=[('ridgecv', regressor)])
  params = [{
        'ridgecv__alphas': alphas,
        'ridgecv__fit_intercept': fit_intercept,
        'ridgecv__scoring': scoring,
        'ridgecv__gcv_mode': gcv_mode,
        'ridgecv__store_cv_values': store_cv_values,
        'ridgecv__alpha_per_target': alpha_per_target,
    }]
  return pipeline, params

# LINEAR STOCHASTIC GRADIENT DESCENT (SGD)
def pipeBuild_SGDRegressor(loss=['squared_error'],penalty=['l2'], alpha=[0.0001],l1_ratio=[0.15],max_iter=[1000],
  fit_intercept=[True],tol=[1.0e-3],shuffle=[True],verbose=[0],epsilon=[0.1],random_state=None,
  learning_rate=['invscaling'],eta0=[0.01],power_t=[0.25],early_stopping=[False],validation_fraction=[0.1],
  warm_start=[False],average=[False]):
  regressor = SGDRegressor(random_state=random_state)
  pipeline = Pipeline(steps=[('lsgd', regressor)])
  params = [{
        'lsgd__loss': loss,
        'lsgd__penalty': penalty,
        'lsgd__alpha': alpha,
        'lsgd__l1_ratio': l1_ratio,
        'lsgd__fit_intercept': fit_intercept,
        'lsgd__max_iter': max_iter,
        'lsgd__tol': tol,
        'lsgd__shuffle': shuffle,
        'lsgd__verbose': verbose,
        'lsgd__epsilon': epsilon,
        'lsgd__learning_rate': learning_rate,
        'lsgd__eta0': eta0,
        'lsgd__power_t': power_t,
        'lsgd__early_stopping': early_stopping,
        'lsgd__validation_fraction': validation_fraction,
        'lsgd__warm_start': warm_start,
        'lsgd__average': average,
    }]
  return pipeline, params

# BAYESIAN ADR
def pipeBuild_ARDRegression(tol=[1.0e-3],alpha_1=[1.0e-6],alpha_2=[1.0e-6], lambda_1=[1.0e-6],
  lambda_2=[1.0e-6],compute_score=[False],threshold_lambda=[10000],fit_intercept=[True],copy_X=[True],
  verbose=[False]):
  regressor = ARDRegression()
  pipeline = Pipeline(steps=[('ard', regressor)])
  params = [{
        'ard__tol': tol,
        'ard__alpha_1': alpha_1,
        'ard__alpha_2': alpha_2,
        'ard__lambda_1': lambda_1,
        'ard__lambda_2': lambda_2,
        'ard__compute_score': compute_score,
        'ard__threshold_lambda': threshold_lambda,
        'ard__fit_intercept': fit_intercept,
        'ard__copy_X': copy_X,
        'ard__verbose': verbose,
    }]
  return pipeline, params

# BAYESIAN RIDGE
def pipeBuild_BayesianRidge(tol=[1.0e-3],alpha_1=[1.0e-6],alpha_2=[1.0e-6], lambda_1=[1.0e-6],
  lambda_2=[1.0e-6],alpha_init=[None],lambda_init=[None],compute_score=[False],fit_intercept=[True],
  copy_X=[True],verbose=[False]):
  regressor = BayesianRidge()
  pipeline = Pipeline(steps=[('bayridge', regressor)])
  params = [{
        'bayridge__tol': tol,
        'bayridge__alpha_1': alpha_1,
        'bayridge__alpha_2': alpha_2,
        'bayridge__lambda_1': lambda_1,
        'bayridge__lambda_2': lambda_2,
        'bayridge__alpha_init': alpha_init,
        'bayridge__lambda_init': lambda_init,
        'bayridge__compute_score': compute_score,
        'bayridge__fit_intercept': fit_intercept,
        'bayridge__copy_X': copy_X,
        'bayridge__verbose': verbose,
    }]
  return pipeline, params

# PASSIVE AGGRESSIVE REGRESSOR
def pipeBuild_PassiveAggressiveRegressor(C=[1.0],fit_intercept=[True],max_iter=[1000],tol=[1.0e-3],
  early_stopping=[False], validation_fraction=[0.1],n_iter_no_change=[5],shuffle=[True],verbose=[0],
  loss=['epsilon_insensitive'],epsilon=[0.1],random_state=None,warm_start=[False],average=[False]):
  regressor = PassiveAggressiveRegressor(random_state=random_state)
  pipeline = Pipeline(steps=[('par', regressor)])
  params = [{
        'par__C': C,        
        'par__fit_intercept': fit_intercept,
        'par__max_iter': max_iter,
        'par__tol': tol,
        'par__early_stopping': early_stopping,
        'par__validation_fraction': validation_fraction,
        'par__n_iter_no_change': n_iter_no_change,
        'par__shuffle': shuffle,
        'par__verbose': verbose,
        'par__loss': loss,        
        'par__epsilon': epsilon,
        'par__warm_start': warm_start,
        'par__average': average,
    }]
  return pipeline, params

# GAMMA REGRESSOR
def pipeBuild_GammaRegressor(alpha=[1.0],fit_intercept=[True], solver=['lbfgs'],max_iter=[100],
  tol=[1.0e-4],warm_start=[False],verbose=[0]):
  regressor = GammaRegressor()
  pipeline = Pipeline(steps=[('gamma', regressor)])
  params = [{
        'gamma__alpha': alpha,
        'gamma__fit_intercept': fit_intercept,
        'gamma__solver': solver,
        'gamma__max_iter': max_iter,
        'gamma__tol': tol,
        'gamma__warm_start': warm_start,
        'gamma__verbose': verbose,
    }]
  return pipeline, params

# POISSON REGRESSOR
def pipeBuild_PoissonRegressor(alpha=[1.0],fit_intercept=[True], solver=['lbfgs'],max_iter=[100],
  tol=[1.0e-4],warm_start=[False],verbose=[0]):
  regressor = PoissonRegressor()
  pipeline = Pipeline(steps=[('poisson', regressor)])
  params = [{
        'poisson__alpha': alpha,
        'poisson__fit_intercept': fit_intercept,
        'poisson__solver': solver,
        'poisson__max_iter': max_iter,
        'poisson__tol': tol,
        'poisson__warm_start': warm_start,
        'poisson__verbose': verbose,
    }]
  return pipeline, params

# TWEEDIE REGRESSOR
def pipeBuild_TweedieRegressor(power=[0],alpha=[1.0],fit_intercept=[True],link=['auto'],solver=['lbfgs'],
  max_iter=[100],tol=[1.0e-4],warm_start=[False],verbose=[0]):
  regressor = TweedieRegressor()
  pipeline = Pipeline(steps=[('tweed', regressor)])
  params = [{
        'tweed__power': power,
        'tweed__alpha': alpha,
        'tweed__fit_intercept': fit_intercept,
        'tweed__link': link,
        'tweed__solver': solver,
        'tweed__max_iter': max_iter,
        'tweed__tol': tol,
        'tweed__warm_start': warm_start,
        'tweed__verbose': verbose,
    }]
  return pipeline, params

# HUBER REGRESSOR
def pipeBuild_HuberRegressor(epsilon=[1.35],alpha=[0.0001],fit_intercept=[True],
    max_iter=[100],tol=[1.0e-5],warm_start=[False]):
  regressor = HuberRegressor()
  pipeline = Pipeline(steps=[('huber', regressor)])
  params = [{
        'huber__epsilon': epsilon,
        'huber__alpha': alpha,
        'huber__fit_intercept': fit_intercept,
        'huber__max_iter': max_iter,
        'huber__tol': tol,
        'huber__warm_start': warm_start,
    }]
  return pipeline, params

# QUANTILE REGRESSOR
def pipeBuild_QuantileRegressor(quantile=[0.5],alpha=[1.0],fit_intercept=[True],
    solver=['highs'],solver_options=[None]):
  regressor = QuantileRegressor()
  pipeline = Pipeline(steps=[('quant', regressor)])
  params = [{
        'quant__quantile': quantile,
        'quant__alpha': alpha,
        'quant__fit_intercept': fit_intercept,
        'quant__solver': solver,
        'quant__solver_options': solver_options,
    }]
  return pipeline, params

# RANSAC REGRESSOR
def pipeBuild_RANSACRegressor(estimator=[None],min_samples=[None],residual_threshold=[None],
    is_data_valid=[None],is_model_valid=[None],max_trials=[100],max_skips=[np.inf],stop_n_inliers=[np.inf],
    stop_score=[np.inf],stop_probability=[0.99],loss=['absolute_error'],random_state=None):
  regressor = RANSACRegressor(random_state=random_state)
  pipeline = Pipeline(steps=[('ranscar', regressor)])
  params = [{
        'ranscar__estimator': estimator,
        'ranscar__min_samples': min_samples,
        'ranscar__residual_threshold': residual_threshold,
        'ranscar__is_data_valid': is_data_valid,        
        'ranscar__is_model_valid': is_model_valid,
        'ranscar__max_trials': max_trials,
        'ranscar__max_skips': max_skips,
        'ranscar__stop_n_inliers': stop_n_inliers,
        'ranscar__stop_score': stop_score,
        'ranscar__stop_probability': stop_probability,
        'ranscar__loss': loss,
    }]
  return pipeline, params

# THEILSEN REGRESSOR
def pipeBuild_TheilSenRegressor(fit_intercept=[True],copy_X=[True],max_subpopulation=[1.0e4],
    n_subsamples=[None],max_iter=[300],tol=[1.0e-3],random_state=None,n_jobs=[None],verbose=[False]):
  regressor = TheilSenRegressor(random_state=random_state)
  pipeline = Pipeline(steps=[('thielsen', regressor)])
  params = [{
        'thielsen__fit_intercept': fit_intercept,
        'thielsen__copy_X': copy_X,
        'thielsen__max_subpopulation': max_subpopulation,
        'thielsen__n_subsamples': n_subsamples,
        'thielsen__max_iter': max_iter,
        'thielsen__tol': tol,
        'thielsen__n_jobs': n_jobs,
        'thielsen__verbose': verbose,
    }]
  return pipeline, params

# ELASTICNET
def pipeBuild_ElasticNet(alpha=[1.0],l1_ratio=[0.5],fit_intercept=[True],precompute=[False],max_iter=[1000],
    copy_X=[True],tol=[1.0e-4],warm_start=[False],positive=[False], random_state=None,selection=['cyclic']):
  regressor = ElasticNet(random_state=random_state)
  pipeline = Pipeline(steps=[('elastic', regressor)])
  params = [{
        'elastic__alpha': alpha,
        'elastic__l1_ratio': l1_ratio,
        'elastic__fit_intercept': fit_intercept,
        'elastic__precompute': precompute,
        'elastic__max_iter': max_iter,
        'elastic__copy_X': copy_X,        
        'elastic__tol': tol,
        'elastic__warm_start': warm_start,
        'elastic__positive': positive,
        'elastic__selection': selection,
    }]
  return pipeline, params

# ELASTICNET CV
def pipeBuild_ElasticNetCV(l1_ratio=[0.5], eps=[0.001], n_alphas=[100], alphas=[None], fit_intercept=[True], 
                           precompute=['auto'], max_iter=[1000], tol=[0.0001], cv=[None], copy_X=[True], 
                           verbose=[0], n_jobs=[None], positive=[False], random_state=None, selection=['cyclic']):
  regressor = ElasticNetCV(random_state=random_state)
  pipeline = Pipeline(steps=[('elasticcv', regressor)])
  params = [{
        'elasticcv__l1_ratio': l1_ratio,
        'elasticcv__eps': eps,
        'elasticcv__n_alphas': n_alphas,
        'elasticcv__alphas': alphas,
        'elasticcv__fit_intercept': fit_intercept,
        'elasticcv__precompute': precompute,
        'elasticcv__max_iter': max_iter,               
        'elasticcv__tol': tol,
        'elasticcv__cv': cv,
        'elasticcv__copy_X': copy_X, 
        'elasticcv__verbose': verbose,
        'elasticcv__n_jobs': n_jobs,
        'elasticcv__positive': positive,
        'elasticcv__selection': selection,
    }]
  return pipeline, params

# MULTITASK ELASTICNET
def pipeBuild_MultiTaskElasticNet(alpha=[1.0], l1_ratio=[0.5], fit_intercept=[True], copy_X=[True], 
                                  max_iter=[1000], tol=[0.0001], warm_start=[False], 
                                  random_state=None, selection=['cyclic']):
  regressor = MultiTaskElasticNet(random_state=random_state)
  pipeline = Pipeline(steps=[('mten', regressor)])
  params = [{
        'mten__alpha': alpha,
        'mten__l1_ratio': l1_ratio,
        'mten__fit_intercept': fit_intercept,
        'mten__copy_X': copy_X,
        'mten__max_iter': max_iter,
        'mten__tol': tol,
        'mten__warm_start': warm_start,
        'mten__selection': selection,
    }]
  return pipeline, params

# MULTITASK ELASTICNET CV
def pipeBuild_MultiTaskElasticNetCV(l1_ratio=[0.5], eps=[0.001], n_alphas=[100], alphas=[None], 
                                    fit_intercept=[True], max_iter=[1000], tol=[0.0001], cv=[None], 
                                    copy_X=[True], verbose=[0], n_jobs=[None], random_state=None, 
                                    selection=['cyclic']):
  regressor = MultiTaskElasticNetCV(random_state=random_state)
  pipeline = Pipeline(steps=[('mtencv', regressor)])
  params = [{        
        'mtencv__l1_ratio': l1_ratio,
        'mtencv__eps': eps,
        'mtencv__n_alphas': n_alphas,
        'mtencv__alphas': alphas,
        'mtencv__fit_intercept': fit_intercept,
        'mtencv__max_iter': max_iter,
        'mtencv__tol': tol,     
        'mtencv__copy_X': copy_X, 
        'mtencv__verbose': verbose,
        'mtencv__n_jobs': n_jobs,
        'mtencv__selection': selection,
    }]
  return pipeline, params

# LARS
def pipeBuild_Lars(fit_intercept=[True],verbose=[False],precompute=['auto'],
    n_nonzero_coefs=[500],eps=[np.finfo(float).eps],copy_X=[True], fit_path=[True],jitter=[None],
    random_state=None):    
  regressor = Lars(random_state=random_state)
  pipeline = Pipeline(steps=[('lars', regressor)])
  params = [{
        'lars__fit_intercept': fit_intercept,
        'lars__verbose': verbose,
        'lars__precompute': precompute,
        'lars__n_nonzero_coefs':n_nonzero_coefs,
        'lars__eps': eps, 
        'lars__copy_X': copy_X,        
        'lars__fit_path': fit_path,
        'lars__jitter': jitter,
    }]
  return pipeline, params

# LARS CV
def pipeBuild_LarsCV(fit_intercept=[True], verbose=[False], max_iter=[500], normalize=['deprecated'], 
                     precompute=['auto'], cv=[None], max_n_alphas=[1000], n_jobs=[None], 
                     eps=[2.220446049250313e-16], copy_X=[True]):    
  regressor = LarsCV()
  pipeline = Pipeline(steps=[('larscv', regressor)])
  params = [{
        'larscv__fit_intercept': fit_intercept,
        'larscv__verbose': verbose,
        'larscv__max_iter': max_iter,        
        'larscv__normalize':normalize,
        'larscv__precompute': precompute,
        'larscv__cv': cv,
        'larscv__max_n_alphas': max_n_alphas,
        'larscv__n_jobs': n_jobs,
        'larscv__eps': eps, 
        'larscv__copy_X': copy_X,
    }]
  return pipeline, params

# LASSO
def pipeBuild_Lasso(alpha=[1.0],fit_intercept=[True],precompute=[False],max_iter=[1000],
    copy_X=[True],tol=[1.0e-4],warm_start=[False],positive=[False], random_state=None,selection=['cyclic']):
  regressor = Lasso(random_state=random_state)
  pipeline = Pipeline(steps=[('lasso', regressor)])
  params = [{
        'lasso__alpha': alpha,
        'lasso__fit_intercept': fit_intercept,
        'lasso__precompute': precompute,
        'lasso__max_iter': max_iter,
        'lasso__copy_X': copy_X,        
        'lasso__tol': tol,
        'lasso__warm_start': warm_start,
        'lasso__positive': positive,
        'lasso__selection': selection,
    }]
  return pipeline, params

# LASSO CV
def pipeBuild_LassoCV(eps=[0.001], n_alphas=[100], alphas=[None], fit_intercept=[True], precompute=['auto'], 
                      max_iter=[1000], tol=[0.0001], copy_X=[True], cv=[None], verbose=[False], n_jobs=[None], 
                      positive=[False], random_state=None, selection=['cyclic']):
  regressor = LassoCV(random_state=random_state)
  pipeline = Pipeline(steps=[('lassocv', regressor)])
  params = [{
        'lassocv__eps': eps,
        'lassocv__n_alphas': n_alphas,
        'lassocv__alphas': alphas,
        'lassocv__fit_intercept': fit_intercept,
        'lassocv__precompute': precompute,
        'lassocv__max_iter': max_iter,
        'lassocv__tol': tol,
        'lassocv__copy_X': copy_X,        
        'lassocv__cv': cv,
        'lassocv__verbose': verbose,
        'lassocv__n_jobs': n_jobs,
        'lassocv__positive': positive,
        'lassocv__selection': selection,
    }]
  return pipeline, params

# LASSO LARS
def pipeBuild_LassoLars(alpha=[1.0], *, fit_intercept=[True], verbose=[False], normalize=['deprecated'], 
                        precompute=['auto'], max_iter=[500], eps=[2.220446049250313e-16], copy_X=[True], 
                        fit_path=[True], positive=[False], jitter=[None], random_state=None):
  regressor = LassoLars(random_state=random_state)
  pipeline = Pipeline(steps=[('lassolars', regressor)])
  params = [{
        'lassolars__alpha': alpha,
        'lassolars__fit_intercept': fit_intercept,
        'lassolars__verbose': verbose,
        'lassolars__normalize': normalize,
        'lassolars__precompute': precompute,
        'lassolars__max_iter': max_iter,
        'lassolars__eps': eps,
        'lassolars__copy_X': copy_X,        
        'lassolars__fit_path': fit_path,
        'lassolars__positive': positive,
        'lassolars__jitter': jitter,
    }]
  return pipeline, params

# LASSO LARS CV
def pipeBuild_LassoLarsCV(fit_intercept=[True], verbose=[False], max_iter=[500], normalize=['deprecated'], 
                          precompute=['auto'], cv=[None], max_n_alphas=[1000], n_jobs=[None], 
                          eps=[2.220446049250313e-16], copy_X=[True], positive=[False]):
  regressor = LassoLarsCV()
  pipeline = Pipeline(steps=[('lassolarscv', regressor)])
  params = [{        
        'lassolarscv__fit_intercept': fit_intercept,
        'lassolarscv__verbose': verbose,
        'lassolarscv__max_iter': max_iter,
        'lassolarscv__normalize': normalize,
        'lassolarscv__precompute': precompute,
        'lassolarscv__cv': cv,
        'lassolarscv__max_n_alphas': max_n_alphas,
        'lassolarscv__n_jobs': n_jobs,
        'lassolarscv__eps': eps,
        'lassolarscv__copy_X': copy_X,
        'lassolarscv__positive': positive,
    }]
  return pipeline, params

# LASSO LARS IC
def pipeBuild_LassoLarsIC(criterion=['aic'], *, fit_intercept=[True], verbose=[False], normalize=['deprecated'], 
                          precompute=['auto'], max_iter=[500], eps=[2.220446049250313e-16], copy_X=[True], 
                          positive=[False], noise_variance=[None]):
  regressor = LassoLarsIC()
  pipeline = Pipeline(steps=[('lassolarsic', regressor)])
  params = [{        
        'lassolarsic__criterion': criterion,
        'lassolarsic__fit_intercept': fit_intercept,
        'lassolarsic__verbose': verbose,        
        'lassolarsic__normalize': normalize,
        'lassolarsic__precompute': precompute,
        'lassolarsic__max_iter': max_iter,
        'lassolarsic__eps': eps,
        'lassolarsic__copy_X': copy_X,
        'lassolarsic__positive': positive,
        'lassolarsic__noise_variance': noise_variance,
    }]
  return pipeline, params

# PERCEPTRON
def pipeBuild_Perceptron(penalty=[None], alpha=[0.0001], l1_ratio=[0.15], fit_intercept=[True], max_iter=[1000], 
                         tol=[0.001], shuffle=[True], verbose=[0], eta0=[1.0], n_jobs=[None], random_state=0, 
                         early_stopping=[False], validation_fraction=[0.1], n_iter_no_change=[5],
                         class_weight=[None], warm_start=[False]):
  regressor = Perceptron(random_state=random_state)
  pipeline = Pipeline(steps=[('perceptron', regressor)])
  params = [{
        'perceptron__penalty': penalty,
        'perceptron__alpha': alpha,
        'perceptron__l1_ratio': l1_ratio,
        'perceptron__fit_intercept': fit_intercept,
        'perceptron__max_iter': max_iter,
        'perceptron__tol': tol,
        'perceptron__shuffle': shuffle,
        'perceptron__verbose': verbose,                
        'perceptron__tol': tol,
        'perceptron__eta0': eta0,
        'perceptron__n_jobs': n_jobs,
        'perceptron__early_stopping': early_stopping,
        'perceptron__validation_fraction': validation_fraction,
        'perceptron__n_iter_no_change': n_iter_no_change,
        'perceptron__class_weight': class_weight,
        'perceptron__warm_start': warm_start,
    }]
  return pipeline, params

# KNN REGRESSOR (TS LEARN)
def pipeBuild_KNeighborsTimeSeriesRegressor(n_neighbors=[5], weights=['uniform'], metric=['dtw'],
                                            metric_params=[None], n_jobs=[None], verbose=[0]):
  regressor = KNeighborsTimeSeriesRegressor()
  pipeline = Pipeline(steps=[('tsknnreg', regressor)])
  params = [{
        'tsknnreg__n_neighbors': n_neighbors,
        'tsknnreg__weights': weights,
        'tsknnreg__metric': metric,
        'tsknnreg__metric_params': metric_params,
        'tsknnreg__n_jobs': n_jobs,        
        'tsknnreg__verbose': verbose,
    }]
  return pipeline, params

# SUPPORT VECTOR MACHINE REGRESSOR (TS LEARN)
def pipeBuild_TimeSeriesSVR(C=[1.0], kernel=['gak'], degree=[3], gamma=['auto'], coef0=[0.0], tol=[0.001],
                            epsilon=[0.1], shrinking=[True], cache_size=[200], n_jobs=[None],
                            verbose=[0], max_iter=[-1]):
  regressor = TimeSeriesSVR()
  pipeline = Pipeline(steps=[('tssvr', regressor)])
  params = [{
        'tssvr__C': C,
        'tssvr__kernel': kernel,
        'tssvr__degree': degree,
        'tssvr__gamma': gamma,
        'tssvr__coef0': coef0,        
        'tssvr__tol': tol,
        'tssvr__epsilon': epsilon,
        'tssvr__shrinking': shrinking,
        'tssvr__cache_size': cache_size,
        'tssvr__n_jobs': n_jobs,              
        'tssvr__verbose': verbose,
        'tssvr__max_iter': max_iter, 
    }]
  return pipeline, params

# ORTHOGANAL MATCHING PURSUIT
def pipeBuild_OrthogonalMatchingPursuit(n_nonzero_coefs=[None], tol=[None], fit_intercept=[True], 
                                        normalize=['deprecated'], precompute=['auto']):
  regressor = OrthogonalMatchingPursuit()
  pipeline = Pipeline(steps=[('omp', regressor)])
  params = [{
        'omp__n_nonzero_coefs': n_nonzero_coefs,
        'omp__tol': tol,
        'omp__fit_intercept': fit_intercept,
        'omp__normalize': normalize,
        'omp__precompute': precompute,
    }]
  return pipeline, params

# ORTHOGANAL MATCHING PURSUIT CV
def pipeBuild_OrthogonalMatchingPursuitCV(copy=[True], fit_intercept=[True], 
                                          max_iter=[None], cv=[None], n_jobs=[None], verbose=[False]):
  regressor = OrthogonalMatchingPursuitCV()
  pipeline = Pipeline(steps=[('ompcv', regressor)])
  params = [{
        'ompcv__copy': copy,        
        'ompcv__fit_intercept': fit_intercept,
        'ompcv__max_iter':  max_iter,
        'ompcv__cv': cv,
        'ompcv__n_jobs': n_jobs,
        'ompcv__verbose': verbose,
    }]
  return pipeline, params

if __name__ == '__main__':
  p = Path('.')
  datapath = p / "test_data/"

  #print("Please enter the file name.  Data files are located in the test_data folder")
  #f_name = input()
  #print(f_name," has been selected")
  
  if(len(sys.argv) <= 1):
    progname = sys.argv[0]
    print(f"Usage: python3 {progname} xxx.npy")
    print(f"Example: python3 {progname} test_data/synthetic_dataset.npy")
    quit()

  file_name = datapath / sys.argv[1]

  #data = ld.load(file_name)
  data = np.load(file_name)

  print("shape of  data is ",data.shape)

  x = data[:, :data.shape[1]-1]  # data
  y = data[:, -1] # label

  n_classes = int(np.amax(y)+1)
  print("number of classes is ",n_classes)

  print("Test array for NaN...",np.isnan(np.min(x)))

  x_axis = np.arange(len(x[0]))

  plot = go.Figure()
  plot.add_trace(go.Scatter(x=x_axis,y=x[0,:]))
  plot.add_trace(go.Scatter(x=x_axis,y=x[1,:]))
  plot.add_trace(go.Scatter(x=x_axis,y=x[2,:]))
  plot.update_layout(title="Data: 1st three samples")
  plot.show()

  # Normalize Data
  x = (x - x.mean(axis=0)) / x.std(axis=0)

  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

  print("Please select the Regression Algorithm you wish to run")
  print("Algorithm List: ",algo_list)
  algo_name = input()
  print("The selected algorithm is: ",algo_name)

  names = []
  pipes = []

  if algo_name == 'svr':
    svr = pipeBuild_SVR()
    names.append('svr')
    pipes.append(svr)
  elif algo_name == 'nusvr':
    nusvr = pipeBuild_NuSVR()
    names.append('nusvr')
    pipes.append(nusvr)
  elif algo_name == 'linear svr':
    linsvr = pipeBuild_LinearSVR()
    names.append('linear svr')
    pipes.append(linsvr)
  elif algo_name == 'ridge':
    ridge = pipeBuild_Ridge()
    names.append('ridge')
    pipes.append(ridge)
  elif algo_name == 'ridge cv':
    ridgecv = pipeBuild_RidgeCV()
    names.append('ridge cv')
    pipes.append(ridgecv)
  elif algo_name == 'linear regression':
    linreg = pipeBuild_LinearRegression()
    names.append('linear regression')
    pipes.append(linreg)
  elif algo_name == 'sgd':
    sgd = pipeBuild_SGDRegressor()
    names.append('sgd')
    pipes.append(sgd)
  elif algo_name == 'ard':
    ard = pipeBuild_ARDRegression()
    names.append('ard')
    pipes.append(ard)
  elif algo_name == 'bayesian ridge':
    bayridge = pipeBuild_BayesianRidge()
    names.append('bayesian ridge')
    pipes.append(bayridge)
  elif algo_name == 'passive aggressive':
    par = pipeBuild_PassiveAggressiveRegressor()
    names.append('passive aggressive')
    pipes.append(par)
  elif algo_name == 'gamma':
    gamma = pipeBuild_GammaRegressor()
    names.append('gamma')
    pipes.append(gamma)
  elif algo_name == 'poisson':
    poisson = pipeBuild_PoissonRegressor()
    names.append('poisson')
    pipes.append(poisson)
  elif algo_name == 'tweedie':
    tweedie = pipeBuild_TweedieRegressor()
    names.append('tweedie')
    pipes.append(tweedie)
  elif algo_name == 'huber':
    huber = pipeBuild_HuberRegressor()
    names.append('huber')
    pipes.append(huber)
  elif algo_name == 'quantile':
    quantile = pipeBuild_QuantileRegressor()
    names.append('quantile')
    pipes.append(quantile)
  elif algo_name == 'ranscar':
    ranscar = pipeBuild_RANSACRegressor()
    names.append('ranscar')
    pipes.append(ranscar)
  elif algo_name == 'thielsen':
    thielsen = pipeBuild_TheilSenRegressor()
    names.append('thielsen')
    pipes.append(thielsen)
  elif algo_name == 'elasticnet':
    elasticnet = pipeBuild_ElasticNet()
    names.append('elasticnet')
    pipes.append(elasticnet)
  elif algo_name == 'elasticnet cv':
    elasticnetcv = pipeBuild_ElasticNetCV()
    names.append('elasticnet cv')
    pipes.append(elasticnetcv)
  elif algo_name == 'lars':
    lars = pipeBuild_Lars()
    names.append('lars')
    pipes.append(lars)  
  elif algo_name == 'lars cv':
    larscv = pipeBuild_LarsCV()
    names.append('lars cv')
    pipes.append(larscv)
  elif algo_name == 'lasso':
    lasso = pipeBuild_Lasso()
    names.append('lasso')
    pipes.append(lasso)
  elif algo_name == 'lasso cv':
    lassocv = pipeBuild_LassoCV()
    names.append('lasso cv')
    pipes.append(lassocv)
  elif algo_name == 'lasso lars':
    lassolars = pipeBuild_LassoLars()
    names.append('lasso lars')
    pipes.append(lassolars)
  elif algo_name == 'lasso lars cv':
    lassolarscv = pipeBuild_LassoLarsCV()
    names.append('lasso lars cv')
    pipes.append(lassolarscv)
  elif algo_name == 'lasso lars ic':
    lassolarsic = pipeBuild_LassoLarsIC()
    names.append('lasso lars ic')
    pipes.append(lassolarsic)
  elif algo_name == 'ts knn':
    tsknn = pipeBuild_TimeSeriesSVR(n_clusters=[n_classes])
    names.append('ts knn')
    pipes.append(tsknn)
  elif algo_name == 'ts svr':
    tssvr = pipeBuild_KNeighborsTimeSeriesRegressor()
    names.append('ts svr')
    pipes.append(tssvr)
  elif algo_name == 'perceptron':
    perceptron = pipeBuild_Perceptron()
    names.append('perceptron')
    pipes.append(perceptron)
  elif algo_name == 'orthogonal matching pursuit':
    omp = pipeBuild_OrthogonalMatchingPursuit()
    names.append('orthogonal matching pursuit')
    pipes.append(omp)
  elif algo_name == 'orthogonal matching pursuit cv':
    ompcv = pipeBuild_OrthogonalMatchingPursuitCV()
    names.append('orthogonal matching pursuit cv')
    pipes.append(ompcv)
  elif algo_name == 'multitask elastic net':
    mten = pipeBuild_MultiTaskElasticNet()
    names.append('multitask elastic net')
    pipes.append(mten)
  elif algo_name == 'multitask elastic net cv':
    mtencv = pipeBuild_MultiTaskElasticNetCV()
    names.append('multitask elastic net cv')
    pipes.append(mtencv)
  else:
    print("You have entered an incorrect algorithm name.  Please rerun the program and select an algoritm from the list")
    exit

  # iterate over regressors
  for j in range(len(names)):
      fig = make_subplots(rows=n_classes, cols=2)

      grid_search = GridSearchCV(estimator=pipes[j][0], param_grid=pipes[j][1], scoring='neg_mean_squared_error',cv=5, verbose=1, n_jobs=-1)
      grid_search.fit(X_train, y_train)
      score = grid_search.score(X_test, y_test)
      print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
      print(grid_search.best_params_)
      y_pred = grid_search.predict(X_test)
      PredictionErrorDisplay.from_estimator(grid_search, X_test, y_test)
      best_title = 'Best Model: ' + names[j]
      plt.title(best_title) 
      
  plt.tight_layout()
  plt.show()