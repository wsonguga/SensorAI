import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import re
import pytz
from datetime import datetime

import enum
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from anomaly_detection import sst_class as sst

import tutorials_old.load_data as ld

algo_list = ['one class svm','sgd one class svm','sst','lof novelty','lof outlier','elliptic envelope','isolation forest']

# All inputs execpt random_state should be lists of values, even if only one value

## NOVELTY DETECTION

# ONCE CLASS SVM
def pipeBuild_OneClassSVM(kernel=['rbf'],degree=[3], gamma=['scale'], coef0=[0.0], tol=[1.0e-3],
                 nu=[0.5], shrinking=[True], cache_size=[200], verbose=[False],max_iter=[-1]):
  classifier = OneClassSVM()
  pipeline = Pipeline(steps=[('1svm', classifier)])
  params = [{
        '1svm__kernel': kernel,
        '1svm__degree': degree,
        '1svm__gamma': gamma,
        '1svm__coef0': coef0,
        '1svm__tol': tol,
        '1svm__nu': nu,
        '1svm__shrinking': shrinking,
        '1svm__cache_size': cache_size,
        '1svm__verbose': verbose,
        '1svm__max_iter': max_iter,
    }]
  return pipeline, params

# SGC ONCE CLASS SVM
def pipeBuild_SGDOneClassSVM(nu=[0.5],fit_intercept=[True], max_iter=[1000], tol=[1.0e-3],
                 shuffle=[True],verbose=[False],random_state=None,learning_rate=['optimal'],
                 eta0=[0.0],power_t=[0.5],warm_start=[False],average=[False]):
  classifier = SGDOneClassSVM(random_state=random_state)
  pipeline = Pipeline(steps=[('sgd1svm', classifier)])
  params = [{
        'sgd1svm__nu': nu,
        'sgd1svm__fit_intercept': fit_intercept,
        'sgd1svm__max_iter': max_iter,        
        'sgd1svm__tol': tol,        
        'sgd1svm__shuffle': shuffle,
        'sgd1svm__verbose': verbose,
        'sgd1svm__learning_rate': learning_rate,
        'sgd1svm__eta0': eta0,
        'sgd1svm__power_t': power_t,
        'sgd1svm__warm_start': warm_start,
        'sgd1svm__average': average,
    }]
  return pipeline, params

# SST ANOMALY DETECTOR
def pipeBuild_SstDetector(y,win_length,order=[None], n_components=[5],lag=[None],
                 is_scaled=[False], use_lanczos=[True], rank_lanczos=[None], eps=[1e-3]):
  detector = sst.SstDetector(y=y,win_length=win_length,order=order,n_components=n_components,
                               lag=lag,is_scaled=is_scaled,use_lanczos=use_lanczos,rank_lanczos=rank_lanczos,eps=eps)
  pipeline = Pipeline(steps=[('sst', detector)])
  params = [{
        #'sst__threshold': threshold,
        'sst__order': order,
        'sst__n_components': n_components,
        'sst__lag': lag,
        'sst__is_scaled': is_scaled,
        'sst__use_lanczos': use_lanczos,
        'sst__rank_lanczos': rank_lanczos,
        'sst__eps': eps,
    }]
  return pipeline, params


# NOVELTY OR OUTLIER DETECTION

# Local Outlier Factor
    # Set novelty to True for novelty detection, otherwise False is outlier detection
def pipeBuild_LocalOutlierFactor(n_neighbors=[20],algorithm=['auto'], leaf_size=[30], metric=['minkowski'],
                 p=[2],verbose=[False],metric_params=[None],contamination=['auto'],
                 novelty=[False],n_jobs=[None]):
  detector = LocalOutlierFactor()
  pipeline = Pipeline(steps=[('lof', detector)])
  params = [{
        'lof__n_neighbors': n_neighbors,
        'lof__algorithm': algorithm,
        'lof__leaf_size': leaf_size,        
        'lof__metric': metric,        
        'lof__p': p,
        'lof__metric_params': metric_params,
        'lof__contamination': contamination,
        'lof__novelty': novelty,
        'lof__n_jobs': n_jobs,
    }]
  return pipeline, params

## OUTLIER DETECTION

# Elliptic Envelope
def pipeBuild_EllipticEnvelope(store_precision=[True],assume_centered=[False], support_fraction=[None], 
                               contamination=[0.1], random_state=None):
  detector = EllipticEnvelope(random_state=random_state)
  pipeline = Pipeline(steps=[('ellenv', detector)])
  params = [{
        'ellenv__store_precision': store_precision,
        'ellenv__assume_centered': assume_centered,
        'ellenv__support_fraction': support_fraction,        
        'ellenv__contamination': contamination,        
    }]
  return pipeline, params

# Isolation Forest
def pipeBuild_IsolationForest(n_estimators=[100],max_samples=['auto'], contamination=['auto'], 
                               max_features=[1.0], bootstrap=[False], n_jobs=[None], random_state=None,
                               verbose=[0],warm_start=[False]):
  detector = IsolationForest(random_state=random_state)
  pipeline = Pipeline(steps=[('isofrst', detector)])
  params = [{
        'isofrst__n_estimators': n_estimators,
        'isofrst__max_samples': max_samples,
        'isofrst__contamination': contamination, 
        'isofrst__max_features': max_features,
        'isofrst__bootstrap': bootstrap,
        'isofrst__n_jobs': n_jobs, 
        'isofrst__verbose': verbose, 
        'isofrst__warm_start': warm_start,
       
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

  data = ld.load(file_name)
  #data = np.load(file_name)

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

  if algo_name == 'sst':
    sst = pipeBuild_SstDetector(y = y_train, win_length = 30, order=[2,5,10,15,20], is_scaled = [True], lag=[2,5,10,15,20])
    #sst = pipeBuild_SstDetector(y = y_train, win_length = 20, order=[10], is_scaled = [True], lag=[10])
    #threshold=[0.1,0.75,1.0,5.0,10.0,50.0], 
    names.append('sst')
    pipes.append(sst)
  elif algo_name == 'one class svm':
    onesvm = pipeBuild_OneClassSVM()
    names.append('one class svm')
    pipes.append(onesvm)
  elif algo_name == 'sgd one class svm':
    sgd1svm = pipeBuild_SGDOneClassSVM()
    names.append('sgd one class svm')
    pipes.append(sgd1svm)
  elif algo_name == 'lof novelty':
    lofn = pipeBuild_LocalOutlierFactor(novelty=[True])
    names.append('lof novelty')
    pipes.append(lofn)
  elif algo_name == 'lof outlier':
    lofo = pipeBuild_LocalOutlierFactor(novelty=[False])
    names.append('lof outlier')
    pipes.append(lofo)
  elif algo_name == 'elliptic envelope':
    ellipenv = pipeBuild_EllipticEnvelope()
    names.append('elliptic envelope')
    pipes.append(ellipenv)
  elif algo_name == 'isolation forest':
    sgd = pipeBuild_IsolationForest()
    names.append('isolation forest')
    pipes.append(sgd)  
  else:
    print("You have entered an incorrect algorithm name.  Please rerun the program and select an algoritm from the list")
    exit

  # iterate over classifiers
  for j in range(len(names)):
      fig = make_subplots(rows=n_classes, cols=2)

      grid_search = GridSearchCV(estimator=pipes[j][0], param_grid=pipes[j][1], scoring='neg_mean_squared_error',cv=5, verbose=1, n_jobs=-1)
      grid_search.fit(X_train, y_train)
      score = grid_search.score(X_test, y_test)
      print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
      print(grid_search.best_params_)
      y_pred = grid_search.predict(X_test)
      print(classification_report(y_test, y_pred))
      #ConfusionMatrixDisplay.from_estimator(grid_search, X_test, y_test, xticks_rotation="vertical")
      plt.title(names[j]+" Heat Map")
      #fig0 = py.plot_mpl(temp.gcf())
      #fig0.show()
      
      count = 0
      while count < len(y_pred):
          fig.add_trace(
              go.Scatter(x=x_axis,y=X_test[count]),
              row=int(y_pred[count])+1, col=1
          )
          fig.add_trace(
              go.Scatter(x=x_axis, y=X_test[count]),
              row=int(y_test[count])+1, col=2
          )
          count = count + 1
      fig.update_layout(title_text = names[j]+": Predicted vs Truth")
      f = 0
      while f < n_classes:
          fig.update_xaxes(title_text="Class "+str(f), row=f+1, col=1)
          fig.update_xaxes(title_text="Class "+str(f), row=f+1, col=2)
          f = f + 1
      fig.show()
  plt.tight_layout()
  plt.show()
"""      best_title = 'Best Model: ' + names[j]
      plt.title(best_title) 
      
  plt.tight_layout()
  plt.show()"""