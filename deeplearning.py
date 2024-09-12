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
from torch.nn.parallel import DataParallel
#from torch.utils.data import DataLoader
from skorch import NeuralNetClassifier
from skorch.helper import SliceDict, SliceDataset
from skorch.callbacks import ProgressBar
from pytorch_weight_norm import WeightNorm
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
#from scikeras.wrappers import KerasClassifier, KerasRegressor
import keras
from keras import Sequential
from keras import layers

#import load_data as ld

device = "cuda" if torch.cuda.is_available() else "cpu"

algo_list = ['lstm','tcn','transformer','sequential']


# Custome SKLEARN Transformer to convert data to PyTorch format
class Slicer(BaseEstimator,TransformerMixin):
   def fit(self,X,y=None):
      return
   
   def transform(self,X,y=None):
      X["sliced data"] = SliceDataset(X)
      return X
   
class ToTensor(BaseEstimator,TransformerMixin):
   def fit(self,X,y=None):
      return
   
   def transform(self,X,y=None):
      X = torch.from_numpy(X)
      return X

# Define Time Series Classification Transformer
# https://keras.io/examples/timeseries/timeseries_classification_transformer/  
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

# TCN with Skorch
def pipeBuild_TCN(num_inputs,num_channels,kernel_size=[4],dilations=[None],
                  dilation_reset=[None],dropout=[0.1],causal=[True],use_norm=[None],
                  activation=['relu'],kernel_initializer=['xavier_uniform'],use_skip_connections=[False],
                  input_shape=['NCL'],embedding_shapes=[None],embedding_mode=['add'],use_gate=[False],
                  lookahead=[1],output_projection=[None],output_activation=[None],epochs=20,lr=0.1): 
    
    tcn = TCN(num_inputs,num_channels)
    #tcn = TCN()

    #cb = ProgressBar()
    classifier = NeuralNetClassifier(
        tcn,
        max_epochs=epochs,
        lr=lr,
        device=device,
        train_split=False,
        verbose=0,
        batch_first=True,
        #callbacks=[cb],
    )
    
    #classifier = DataParallel(classifier, device_ids=[0]) 

    #_ = pickle.dumps(tcn)  # raises Exception
    #del cb.pbar
    #_ = pickle.dumps(tcn)  # works

    #pipeline = Pipeline(steps=[('data convert',Slicer()),('tcn', classifier)])
    #pipeline = Pipeline(steps=[('tensor data',ToTensor()),('tcn', classifier)])
    pipeline = Pipeline(steps=[('tcn', classifier)])

    params = [{
        #'tcn__num_inputs': num_inputs,
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


# LSTM
# This is a simple model and was included to test SciKeras with SKLearn
def pipeBuild_LSTM(build_fn=[None],warm_start=[False],random_state=[None],optimizer=['rmsprop'],
                         loss=['sparse_categorical_crossentropy'],metrics=[None],batch_size=[None],validation_batch_size=[None],
                         verbose=[1],callbacks=[None],validation_split=[0.0],shuffle=[True],
                         run_eagerly=[False],epochs=[1],class_weight=[None]): 
    
    #def get_model(hidden_layer_dim, meta):
    def get_model(look_back, meta):
        # note that meta is a special argument that will be
        # handed a dict containing input metadata
        n_features_in_ = meta["n_features_in_"]
        X_shape_ = meta["X_shape_"]
        n_classes_ = meta["n_classes_"]

        model = Sequential()
        model.add(LSTM(4, input_shape=(X_shape_[1:], look_back)))
        model.add(Dense(1))
        #model.add(keras.layers.LSTM(n_features_in_, input_shape=X_shape_[1:]))
        #model.add(keras.layers.LSTM(n_features_in_, batch_input_shape=(batch_size, X_shape_[1:], X_shape_[2:]), stateful=True))
        #model.add(keras.layers.Dense(n_classes_))
        #model.add(keras.layers.Activation("softmax"))
        return model
    
    classifier = KerasClassifier(
        model=get_model,
        loss=loss,
        optimizer=optimizer,
        look_back=2
        #hidden_layer_dim=100,
    )
    
    pipeline = Pipeline(steps=[('lstm', classifier)])

    params = [{
        'lstm__build_fn': build_fn,
        'lstm__warm_start': warm_start,
        'lstm__random_state': random_state,
        'lstm__optimizer': optimizer,
        'lstm__loss': loss,
        'lstm__metrics': metrics,
        'lstm__batch_size': batch_size,
        'lstm__validation_batch_size': validation_batch_size,
        'lstm__verbose': verbose,
        'lstm__callbacks': callbacks,
        'lstm__shuffle': shuffle,
        'lstm__run_eagerly': run_eagerly,
        'lstm__validation_split': validation_split,
        'lstm__epochs': epochs,
        'lstm__class_weight': class_weight,
    }]
    return pipeline, params


# SEQUENTIAL
# This is a simple model and was included to test SciKeras with SKLearn
def pipeBuild_Sequential(build_fn=[None],warm_start=[False],random_state=[None],optimizer=['rmsprop'],
                         loss=['sparse_categorical_crossentropy'],metrics=[None],batch_size=[None],validation_batch_size=[None],
                         verbose=[1],callbacks=[None],validation_split=[0.0],shuffle=[True],
                         run_eagerly=[False],epochs=[1],class_weight=[None]): 
    
    def get_model(hidden_layer_dim, meta):
        # note that meta is a special argument that will be
        # handed a dict containing input metadata
        n_features_in_ = meta["n_features_in_"]
        X_shape_ = meta["X_shape_"]
        n_classes_ = meta["n_classes_"]

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(n_features_in_, input_shape=X_shape_[1:]))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dense(hidden_layer_dim))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dense(n_classes_))
        model.add(keras.layers.Activation("softmax"))
        return model
    
    classifier = KerasClassifier(
        model=get_model,
        loss=loss,
        optimizer=optimizer,
        hidden_layer_dim=100,
    )
    
    pipeline = Pipeline(steps=[('seq', classifier)])

    params = [{
        'seq__build_fn': build_fn,
        'seq__warm_start': warm_start,
        'seq__random_state': random_state,
        'seq__optimizer': optimizer,
        'seq__loss': loss,
        'seq__metrics': metrics,
        'seq__batch_size': batch_size,
        'seq__validation_batch_size': validation_batch_size,
        'seq__verbose': verbose,
        'seq__callbacks': callbacks,
        'seq__shuffle': shuffle,
        'seq__run_eagerly': run_eagerly,
        'seq__validation_split': validation_split,
        'seq__epochs': epochs,
        'seq__class_weight': class_weight,
    }]
    return pipeline, params


# TRANSFORMER
# This is a simple model and was included to test SciKeras with SKLearn
def pipeBuild_Transformer(build_fn=[None],warm_start=[False],random_state=[None],optimizer=['rmsprop'],
                         loss=['sparse_categorical_crossentropy'],metrics=[None],batch_size=[None],validation_batch_size=[None],
                         verbose=[1],callbacks=[None],validation_split=[0.0],shuffle=[True],
                         run_eagerly=[False],epochs=[1],class_weight=[None]): 
    
    def get_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, 
                  dropout=0, mlp_dropout=0,
    ):
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(n_classes, activation="softmax")(x)
        return keras.Model(inputs, outputs)
    
    classifier = KerasClassifier(
        model=get_model,
        loss=loss,
        optimizer=optimizer,
        hidden_layer_dim=100,
    )
    
    pipeline = Pipeline(steps=[('transformer', classifier)])

    params = [{
        'transformer__build_fn': build_fn,
        'transformer__warm_start': warm_start,
        'transformer_random_state': random_state,
        'transformer__optimizer': optimizer,
        'transformer__loss': loss,
        'transformer__metrics': metrics,
        'transformer__batch_size': batch_size,
        'transformer__validation_batch_size': validation_batch_size,
        'transformer__verbose': verbose,
        'transformer__callbacks': callbacks,
        'transformer__shuffle': shuffle,
        'transformer__run_eagerly': run_eagerly,
        'transformer__validation_split': validation_split,
        'transformer__epochs': epochs,
        'transformer__class_weight': class_weight,
    }]
    return pipeline, params




# DEEPLEARNING CLASSIFICATON GIRD BUILDER
def gridsearch_classifier(names,pipes,X_train,X_test,y_train,y_test,scoring='accuracy',plot_number=10):
    # iterate over classifiers
    for j in range(len(names)):

        grid_search = GridSearchCV(estimator=pipes[j][0], param_grid=pipes[j][1], scoring=scoring, refit=True,
                                   cv=5, verbose=1, n_jobs=-1)           
        #X_train = torch.from_numpy(X_train)
        print("algo name is ",names[j])
        if names[j] == 'tcn':
            #X_tensor = SliceDataset(X_train)
            #X_tensor = torch.from_numpy(X_train).detach()
            X_tensor = X_train.tolist()
            grid_search.fit(X_tensor, y_train)
        else:
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
                    if y_pred[count] == current_label and plot_num < plot_number:
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