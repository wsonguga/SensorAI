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
from sklearn.metrics import PredictionErrorDisplay

import utils

def gridsearch_classifier(names,pipes,X_train,X_test,y_train,y_test,scoring='neg_mean_squared_error',plot_number=10):
    # iterate over classifiers
    for j in range(len(names)):

        grid_search = GridSearchCV(estimator=pipes[j][0], param_grid=pipes[j][1], scoring=scoring,cv=5, verbose=1, n_jobs=-1)
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
                        fig.add_trace(
                            go.Scatter(x=x_axis,y=X_test[count]),
                            row=plot_num+1, col=current_label+1
                        )                        
                        plot_num = plot_num +1
                    if y_pred[count] == y_test[count]:
                        color = 'black'
                    else:
                        color = 'red'
                    fig.update_traces(line_color=color)
                    count = count + 1
                current_label = current_label +1
                plot_num = 0
                count = 0
        else:
            print("Incorrect plot number value entered")
        fig.show()
    return

def gridsearch_clustering(names,pipes,X,y,scoring='rand_score',plot_number='all'):
  # iterate over cluterers
  for j in range(len(names)):

        grid_search = GridSearchCV(estimator=pipes[j][0], param_grid=pipes[j][1], scoring=scoring,cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X, y)
        #score = grid_search.score(X, y)
        print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
        print(grid_search.best_params_)
        print("Best "+scoring+"score: ",grid_search.best_score_)
        labels = grid_search.best_estimator_.steps[0][1].labels_
        #print("Best Model Labels: ",labels)
        noise = np.isin(labels, -1)
        if np.any(noise)==True:
            new_noise_label = int(np.amax(labels)+1) # find the max label value
            labels = np.where(labels == -1, new_noise_label, labels)
            

        x_classes = int(np.amax(labels)+1)
        y_classes = int(np.amax(y)+1)
        print("# of X's classes is: ",x_classes)
        print("# of y's classes is: ",y_classes)
        if x_classes > y_classes:
            n_classes = x_classes
        else:
            n_classes = y_classes  
        x_axis = np.arange(len(X[0]))

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
                while count < len(y):
                    if labels[count] == current_label and plot_num < plot_number:
                        fig.add_trace(
                            go.Scatter(x=x_axis,y=X[count]),
                            row=plot_num+1, col=current_label+1
                        )
                        plot_num = plot_num +1
                    count = count + 1
                current_label = current_label +1
                plot_num = 0
                count = 0
        else:
            print("Incorrect plot number value entered")
        fig.show()

def gridsearch_regressor(names,pipes,X_train,X_test,y_train,y_test,scoring='neg_mean_squared_error'):
    # iterate over regressors
    for j in range(len(names)):

        grid_search = GridSearchCV(estimator=pipes[j][0], param_grid=pipes[j][1], scoring=scoring,cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        score = grid_search.score(X_test, y_test)
        print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
        print(grid_search.best_params_)
        y_pred = grid_search.predict(X_test)
        
        plt = utils.plot_2vectors(label=y_test, pred=y_pred, name=names[j], size=10)
        #PredictionErrorDisplay.from_estimator(grid_search, X_test, y_test)
        best_title = 'Best Model: ' + names[j]
        plt.title(best_title)

        plt.show()
    return

def gridsearch_outlier(names,pipes,X,y,scoring='neg_mean_squared_error',plot_number=10):
    # iterate over classifiers
    for j in range(len(names)):

        grid_search = GridSearchCV(estimator=pipes[j][0], param_grid=pipes[j][1], scoring=scoring,cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X, y)
        #score = grid_search.score(X, y)
        print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
        print(grid_search.best_params_)
        #ConfusionMatrixDisplay.from_estimator(grid_search, X, y, xticks_rotation="vertical")
                   
        n_classes = int(np.amax(y)+1) 
        x_axis = np.arange(len(X[0]))
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
                while count < len(y):
                    if y[count] == current_label and plot_num < plot_number:
                        fig.add_trace(
                            go.Scatter(x=x_axis,y=X[count]),
                            row=plot_num+1, col=current_label+1
                        )                        
                        plot_num = plot_num +1
                    count = count + 1
                current_label = current_label +1
                plot_num = 0
                count = 0
        else:
            print("Incorrect plot number value entered")
        fig.show()
    return

""" # Old version
def gridsearch_classifier(names,pipes,X_train,X_test,y_train,y_test,scoring='neg_mean_squared_error',plot_number='all'):
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
                   
        n_classes = int(np.amax(y_test)+1) 
        x_axis = np.arange(len(X_test[0]))
        fig = make_subplots(rows=n_classes, cols=2)

        count = 0
        current_label = 0
        plot_num = 0
        if isinstance(plot_number,int) and plot_number > 0 and plot_number <= 10:
            while current_label < n_classes:
                while count < len(y_test):
                    if y_test[count] == current_label and plot_num < plot_number:
                        fig.add_trace(
                            go.Scatter(x=x_axis,y=X_test[count]),
                            row=int(y_pred[count])+1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=x_axis, y=X_test[count]),
                            row=int(y_test[count])+1, col=2
                        )
                        plot_num = plot_num +1
                    count = count + 1
                current_label = current_label +1
                plot_num = 0
                count = 0
        elif plot_number == 'all': 
            while count < len(y_test):
                fig.add_trace(
                    go.Scatter(x=x_axis,y=X_test[count]),
                    row=int(y_pred[count])+1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=x_axis, y=X_test[count]),
                    row=int(y_test[count])+1, col=2
                )
                count = count + 1
        else:
            print("Incorrect plot number value entered")

        fig.update_layout(title_text = names[j]+": Predicted vs Truth")
        f = 0
        while f < n_classes:
            fig.update_xaxes(title_text="Class "+str(f), row=f+1, col=1)
            fig.update_xaxes(title_text="Class "+str(f), row=f+1, col=2)
            f = f + 1
        fig.show()
    return"""

""" # Old version
def gridsearch_clustering(names,pipes,X,y,scoring='rand_score',plot_number='all'):
  # iterate over cluterers
  for j in range(len(names)):

        grid_search = GridSearchCV(estimator=pipes[j][0], param_grid=pipes[j][1], scoring=scoring,cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X, y)
        #score = grid_search.score(X, y)
        print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
        print(grid_search.best_params_)
        print("Best "+scoring+"score: ",grid_search.best_score_)
        labels = grid_search.best_estimator_.steps[0][1].labels_
        #print("Best Model Labels: ",labels)
        noise = np.isin(labels, -1)
        if np.any(noise)==True:
            new_noise_label = int(np.amax(labels)+1) # find the max label value
            labels = np.where(labels == -1, new_noise_label, labels)
            

        x_classes = int(np.amax(labels)+1)
        y_classes = int(np.amax(y)+1)
        print("# of X's classes is: ",x_classes)
        print("# of y's classes is: ",y_classes)
        if x_classes > y_classes:
            n_classes = x_classes
        else:
            n_classes = y_classes  
        x_axis = np.arange(len(X[0]))
        fig = make_subplots(rows=n_classes, cols=2)

      
        count = 0
        current_label = 0
        plot_num = 0
        if isinstance(plot_number,int) and plot_number > 0 and plot_number <= 10:
            while current_label < n_classes:
                while count < len(y):
                    if y[count] == current_label and plot_num < plot_number:
                        fig.add_trace(
                            go.Scatter(x=x_axis,y=X[count]),
                            row=int(labels[count])+1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=x_axis, y=X[count]),
                            row=int(y[count])+1, col=2
                        )
                        plot_num = plot_num +1
                    count = count + 1
                current_label = current_label +1
                plot_num = 0
                count = 0
        elif plot_number == 'all': 
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
        else:
            print("Incorrect plot number value entered")

        fig.update_layout(title_text = names[j]+": Predicted vs Truth")
        f = 0
        while f < n_classes:
            fig.update_xaxes(title_text="Class "+str(f), row=f+1, col=1)
            fig.update_xaxes(title_text="Class "+str(f), row=f+1, col=2)
            f = f + 1
        fig.show()"""