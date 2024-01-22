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
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, RocCurveDisplay, auc, roc_curve, roc_auc_score

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier, BallTree, KDTree
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neural_network import BernoulliRBM

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import joblib

import tutorials_old.load_data as ld
import sk_classifier_builder as skb
import tutorials_old.sk_classifier_metrics as skm

import sk_novelty_builder as skn

#st.title('Streamlit Demo')

p = Path('.')

save_eval_path = p / "AI_engine/evaluation_results/"
save_model_path = p / "AI_engine/saved_models/"
datapath = p / "AI_engine/test_data/"

#data = np.load("C:/Users/steph/OneDrive/Documents/GitHub/IIoT_Datahub/AI_engine/test_data/synthetic_dataset.npy")
#data = np.load(datapath / "synthetic_dataset.npy")
#data = np.load("C:/Users/steph/OneDrive/Documents/GitHub/IIoT_Datahub/AI_engine/test_data/syn2class.npy")
data = ld.selectFileAndLoad()


print("shape of data is ",data.shape)
#print("NaNs in data? ",np.isnan(np.min(data)))


x = data[:, :data.shape[1]-1]  # data
y = data[:, -1] # label
print("shape of x is ",x.shape)
print("length of x is ",len(x[0]))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)


# Build SST Pipeline.  Working So Long as valid variable combinations are selected for win_lenth, order, and lag. 
sst = skn.pipeBuild_SstDetector(win_length = 20, order=[10], threshold=[0.1,0.75,1.0,5.0,10.0,50.0], is_scaled = [True],lag=[10])

# Build SK Learn Pipelines for Novelty Detection
onesvm = skn.pipeBuild_OneClassSVM(kernel=['rbf','linear'])
sgd1svm = skn.pipeBuild_SGDOneClassSVM(learning_rate=['adaptive','optimal'])
lofn = skn.pipeBuild_LocalOutlierFactor(algorithm=['ball_tree','kd_tree'],novelty=[True])

# Build Sk Learn Pipelines for Outlier Detection
lofo = skn.pipeBuild_LocalOutlierFactor(algorithm=['ball_tree','kd_tree'],novelty=[False])
ellenv = skn.pipeBuild_EllipticEnvelope()
isofrst = skn.pipeBuild_IsolationForest()

# Run All Novelty
#names = ['1 Class SVM','SGD 1 Class SVM','SST','LOF Novelty']
#pipes = [onesvm,sgd1svm,sst,lofn]

# Run All Outlier
#names = ['LOF outlier','Elliptic Envelope']
#pipes = [lofo]

# Run One
names=['Isolation Forest']
pipes=[isofrst]


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

# iterate over classifiers
for j in range(len(names)):
    
    grid_search = GridSearchCV(estimator=pipes[j][0], param_grid=pipes[j][1], scoring='neg_mean_squared_error',cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    score = grid_search.score(X_test, y_test)
    print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
    print(grid_search.best_params_)
    y_pred = grid_search.predict(X_test)
    print(classification_report(y_test, y_pred))
    #ConfusionMatrixDisplay.from_estimator(grid_search, X_test, y_test, xticks_rotation="vertical")

#plt.tight_layout()
#plt.show()
#"""