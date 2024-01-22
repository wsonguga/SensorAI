from math import radians
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from pathlib import Path

#import streamlit as st

import re
import pytz
from datetime import datetime

import enum
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import PredictionErrorDisplay

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import joblib

import tutorials_old.load_data as ld
import sk_regressor_builder as skr


#st.title('Streamlit Demo')

p = Path('.')

save_eval_path = p / "AI_engine/evaluation_results/"
save_model_path = p / "AI_engine/saved_models/"
datapath = p / "AI_engine/test_data/"

#data = np.load("C:/Users/steph/OneDrive/Documents/GitHub/IIoT_Datahub/AI_engine/test_data/synthetic_dataset.npy")
#data = np.load(datapath / "synthetic_dataset.npy")
#data = np.load("C:/Users/steph/OneDrive/Documents/GitHub/IIoT_Datahub/AI_engine/test_data/syn2class.npy")

data = ld.selectFileAndLoad()
# if dataset == None:
#     # Load the Sklearn dataset
#     dataset = datasets.load_diabetes()
#     # Read the DataFrame, first using the feature data
#     df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
#     # Add a target column, and fill it with the target data
#     df['target'] = dataset.target
#     # Convert to Numpy Array
#     data = df.to_numpy().copy()
#     np.save(datapath / 'sk_diabetes.npy', data)
# else:
#     data = dataset.copy()


#"""
print("shape of  data is ",data.shape)

x = data[:, :data.shape[1]-1]  # data
y = data[:, -1] # target

#print("shape of x is ",x.shape)
#print("shape of y is ",y.shape)

# normalization on input data x
x = (x - x.mean(axis=0)) / x.std(axis=0)

# Use line below with PV_Data Only
#x = np.delete(x, 799999, 1)  # delete second column of C

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

#print(x)
#print(X_train)

# SK LEARN REGRESSORS
svr = skr.pipeBuild_SVR(kernel=['linear','rbf'])
nusvr = skr.pipeBuild_NuSVR(kernel=['linear','rbf'])
lsvr = skr.pipeBuild_LinearSVR(loss=['epsilon_insensitive','squared_epsilon_insensitive'])
ridge = skr.pipeBuild_Ridge(alpha=[1.0,2.0])
ridgecv = skr.pipeBuild_RidgeCV(alphas=[(0.1, 1.0, 10.0),(0.2, 2.0, 20.0)])
linreg = skr.pipeBuild_LinearRegression()
sgd = skr.pipeBuild_SGDRegressor(loss=['squared_error','huber'])
ard = skr.pipeBuild_ARDRegression()
bayridge = skr.pipeBuild_BayesianRidge()
par = skr.pipeBuild_PassiveAggressiveRegressor()
gamma = skr.pipeBuild_GammaRegressor(solver=['lbfgs','newton-cholesky'])
poiss = skr.pipeBuild_PoissonRegressor(solver=['lbfgs','newton-cholesky'])
tweed = skr.pipeBuild_TweedieRegressor(solver=['lbfgs','newton-cholesky'])
huber = skr.pipeBuild_HuberRegressor(epsilon=[1.35,2.5])
quant = skr.pipeBuild_QuantileRegressor(solver=['highs','highs-ds'])
ranscar = skr.pipeBuild_RANSACRegressor(loss=['absolute_error','squared_error'])
thielsen = skr.pipeBuild_TheilSenRegressor()
elastic = skr.pipeBuild_ElasticNet(selection=['cyclic','random'])
lars = skr.pipeBuild_Lars()
lasso = skr.pipeBuild_Lasso(selection=['cyclic','random'])

#TS LEARN REGRESSORS
tssvr = skr.pipeBuild_TimeSeriesSVR(kernel=['rbf'])
tsknn = skr.pipeBuild_KNeighborsTimeSeriesRegressor(weights=['distance'])


# Run All
#names = ['SVR','NuSVR','LinearSVR','Ridge','RidgeCV','LinearRegression','SGD','Bayesian ADR',
#   'Bayesian Ridge','Passive Aggressive','Gamma','Poisson','Tweedie','Huber','Quantile','RANSCAR',
#   'ThielSen','ElasticNet','Lars','Lasso','TS KNN','TS SVR']
#pipes = [svr,nusvr,lsvr,ridge,ridgecv,linreg,sgd,ard,bayridge,par,gamma,poiss,tweed,huber,quant,
#   ranscar,thielsen,elastic,lars,lasso,tsknn,tssvr]

# Run Single
names=['TS SVR','TS KNN']
pipes=[tssvr,tsknn]

titles = []
for t in names:
    tn = t + ' Train'
    ts = t + ' Test'
    titles.append(tn)
    titles.append(ts)


#x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
#y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5

samples = np.arange(len(X_train[0,:]))
#print("samples: ",samples)

# Plot Training Set
plt.plot(X_train[0,:]) 
plt.plot(X_train[1,:]) 
plt.plot(X_train[2,:]) 
#fig1 = px.scatter(x = samples,y = X_train[0,:],title="Sample Data Entry")
#st.plotly_chart(fig1)
plt.show()

# Plot Testing Set
#fig1.append_trace(go.Scatter(x = X_test[:, 0],y = X_test[:, 1],),row=i,col=1)

#fig2, ax = plt.subplots(1,len(names))

# iterate over regressors
for j in range(len(names)):
    
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
#st.pyplot(plt)
plt.show()
#fig2.show()
#"""