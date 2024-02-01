from math import radians
import numpy as np
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
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
from sklearn.metrics.cluster import adjusted_rand_score, rand_score, mutual_info_score, normalized_mutual_info_score

from plotly.subplots import make_subplots

def gridsearch_classifier(names,pipes,X_train,X_test,y_train,y_test,scoring='neg_mean_squared_error'):
    # iterate over classifiers
    for j in range(len(names)):

        #today = date.today()
        #now = today.strftime("%b-%d-%Y")
        #save_file = str(names[j]) + '-' + str(now) + '-HeatMap.png'

        grid_search = GridSearchCV(estimator=pipes[j][0], param_grid=pipes[j][1], scoring=scoring,cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        score = grid_search.score(X_test, y_test)
        print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
        print(grid_search.best_params_)
        y_pred = grid_search.predict(X_test)
        print(classification_report(y_test, y_pred))
        ConfusionMatrixDisplay.from_estimator(grid_search, X_test, y_test, xticks_rotation="vertical")
    return

def gridsearch_clustering(names,pipes,X,y,scoring='rand_score'):
  # iterate over cluterers
  for j in range(len(names)):
      #x_classes = int(np.amax(X)+1)
      #y_classes = int(np.amax(y)+1)
      #if x_classes > y_classes:
      #    n_classes = x_classes
      #else:
      #    n_classes = y_classes  
      #x_axis = np.arange(len(X[0]))
      #fig = make_subplots(rows=n_classes, cols=2)

      grid_search = GridSearchCV(estimator=pipes[j][0], param_grid=pipes[j][1], scoring=scoring,cv=5, verbose=1, n_jobs=-1)
      grid_search.fit(X, y)
      #score = grid_search.score(X, y)
      print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
      print(grid_search.best_params_)
      print("Best "+scoring+"score: ",grid_search.best_score_)
      #y_pred = grid_search.predict(X_test)
      #print(classification_report(y_test, y_pred))
      labels = grid_search.best_estimator_.steps[1][1].labels_

      x_classes = int(np.amax(X)+1)
      y_classes = int(np.amax(y)+1)
      if x_classes > y_classes:
        n_classes = x_classes
      else:
        n_classes = y_classes  
      x_axis = np.arange(len(X[0]))
      fig = make_subplots(rows=n_classes, cols=2)

      count = 0
      while count < len(y):
          fig.add_trace(
              go.Scatter(x=x_axis,y=X[count]),
              row=int(labels[count])+1, col=1
          )
          fig.add_trace(
              go.Scatter(x=x_axis, y=X[count]),
              row=int(y[count])+1, col=2
          )
          count = count + 1
      fig.update_layout(title_text = names[j]+": Predicted vs Truth")
      f = 0
      while f < n_classes:
          fig.update_xaxes(title_text="Class "+str(f), row=f+1, col=1)
          fig.update_xaxes(title_text="Class "+str(f), row=f+1, col=2)
          f = f + 1
      fig.show()

def gridsearch_regressor(names,pipes,X_train,X_test,y_train,y_test,scoring='accuracy'):
    # iterate over regressors
    for j in range(len(names)):

        grid_search = GridSearchCV(estimator=pipes[j][0], param_grid=pipes[j][1], scoring=scoring,cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        score = grid_search.score(X_test, y_test)
        print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
        print(grid_search.best_params_)
        y_pred = grid_search.predict(X_test)
        
        plt.scatter(y_pred, y_test)
        plt.show()
    return