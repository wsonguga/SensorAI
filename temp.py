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
    return