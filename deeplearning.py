import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from plotly.subplots import make_subplots
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
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, RocCurveDisplay, auc, roc_curve, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
import torch
from pytorch_tcn import TCN
from skorch import NeuralNetClassifier
from skorch.helper import SliceDict, SliceDataset
from pytorch_weight_norm import WeightNorm

#import load_data as ld

device = "cuda" if torch.cuda.is_available() else "cpu"

algo_list = ['lstm','tcn','transformer']


# Custome SKLEARN Transformer to convert data to PyTorch format
class Slicer(BaseEstimator,TransformerMixin):
   def fit(self,X,y=None):
      return
   
   def transform(self,X,y=None):
      X["sliced data"] = SliceDict(X)
      return X
   
class ToTensor(BaseEstimator,TransformerMixin):
   def fit(self,X,y=None):
      return
   
   def transform(self,X,y=None):
      X = torch.from_numpy(X)
      return X

# TCN
def pipeBuild_TCN(num_inputs,num_channels,kernel_size=[4],dilations=[None],
                  dilation_reset=[None],dropout=[0.1],causal=[True],use_norm=[None],
                  activation=['relu'],kernel_initializer=['xavier_uniform'],use_skip_connections=[False],
                  input_shape=['NCL'],embedding_shapes=[None],embedding_mode=['add'],use_gate=[False],
                  lookahead=[1],output_projection=[None],output_activation=[None],epochs=20,lr=0.1): 
    
    tcn = TCN(num_inputs,num_channels)
    
    classifier = NeuralNetClassifier(
        tcn,
        max_epochs=epochs,
        lr=lr,
        device=device,
        train_split=False,
        verbose=0,
    )
    
    #pipeline = Pipeline(steps=[('data convert',Slicer()),('tcn', classifier)])
    #pipeline = Pipeline(steps=[('tensor data',ToTensor()),('tcn', classifier)])
    pipeline = Pipeline(steps=[('tcn', classifier)])

    params = [{
        'tcn__num_inputs': num_inputs,
        'tcn__num_channels': num_channels,
        'tcn__kernel_size': kernel_size,
        'tcn__dilations': dilations,
        'tcn__dilation_reset': dilation_reset,
        'tcn__dropout': dropout,
        'tcn__causal': causal,
        'tcn__use_norm': use_norm,
        'tcn__activation': activation,
        'tcn__kernel_initializer': kernel_initializer,
        'tcn__kernel_use_skip_connections': use_skip_connections,
        'tcn__input_shape': input_shape,
        'tcn__embedding_shapes': embedding_shapes,
        'tcn__embedding_mode': embedding_mode,
        'tcn__use_gate': use_gate,
        'tcn__lookahead': lookahead,
        'tcn__output_projection': output_projection,
        'tcn__output_activation': output_activation,
    }]
    return pipeline, params



# DEEPLEARNING CLASSIFICATON GIRD BUILDER
def gridsearch_classifier(names,pipes,X_train,X_test,y_train,y_test,scoring='accuracy',plot_number=10):
    # iterate over classifiers
    for j in range(len(names)):

        grid_search = GridSearchCV(estimator=pipes[j][0], param_grid=pipes[j][1], scoring=scoring, refit=False,
                                   cv=5, verbose=1, n_jobs=-1)
        #print("X_train type: ",type(X_train))
        #X_train = SliceDataset(X_train)
        #print("Sliced type: ",type(X_train))
        X_train = torch.from_numpy(X_train)
        grid_search.fit(X_train, y_train)
        score = grid_search.score(X_test, y_test)
        print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
        print(grid_search.best_params_)
        y_pred = grid_search.predict(X_test)
        print(classification_report(y_test, y_pred))
        ConfusionMatrixDisplay.from_estimator(grid_search, X_test, y_test, xticks_rotation="vertical")
                   
        n_classes = int(np.amax(y_test)+1) 
        x_axis = np.arange(len(X_test[0]))
        j = 0
        titles = []
        while j < n_classes:
            name = "Class " + str(j)
            titles.append(name)
            j = j+1
        fig = make_subplots(
            rows=plot_number, cols=n_classes,
            subplot_titles=titles)

        count = 0
        current_label = 0
        plot_num = 0
        if isinstance(plot_number,int) and plot_number > 0 and plot_number <= 10:
            while current_label < n_classes:
                while count < len(y_test):
                    if y_test[count] == current_label and plot_num < plot_number:
                        if y_pred[count] == y_test[count]:
                            color = 'black'
                        else:
                            color = 'red'
                        fig.add_trace(
                            go.Scatter(
                                mode='lines+markers',
                                x=x_axis,
                                y=X_test[count],
                                marker=dict(
                                  color=color,
                                  size = 2,
                                  )),
                            row=plot_num+1, col=current_label+1                            
                        )                       
                        plot_num = plot_num +1
                    count = count + 1
                current_label = current_label +1
                plot_num = 0
                count = 0
        else:
            print("Incorrect plot number value entered")
        fig.update_layout(showlegend=False)
        fig.show()
    return

# MAIN
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

  print("Please select the Classification Algorithm you wish to run")
  print("Algorithm List: ",algo_list)
  algo_name = input()
  print("The selected algorithm is: ",algo_name)

  names = []
  pipes = []

  if algo_name == 'tcn':
    dt = pipeBuild_TCN()
    names.append('tcn')
    pipes.append(dt)
  elif algo_name == 'random forest':
    rf = pipeBuild_RandomForestClassifier()
    names.append('random forest')
    pipes.append(rf)
  elif algo_name == 'knn':
    knn = pipeBuild_KNeighborsClassifier()
    names.append('knn')
    pipes.append(knn)  
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
      ConfusionMatrixDisplay.from_estimator(grid_search, X_test, y_test, xticks_rotation="vertical")
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