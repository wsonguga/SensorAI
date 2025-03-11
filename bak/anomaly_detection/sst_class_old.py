from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, OutlierMixin
from anomaly_detection.fastsst.sst import *

class SstDetector(BaseEstimator, OutlierMixin):

    def __init__(self,  win_length, threshold, order, n_components=5, lag=None,
                 is_scaled=False, use_lanczos=True, rank_lanczos=None, eps=1e-3, **kwargs):
        self.kwargs = kwargs
        # grid search attributes
        self.threshold = threshold
        self.order = order        
        self.eps = eps
        self.is_scaled = is_scaled
        self.n_components = n_components
        self.lag = lag
        self.win_length = win_length
        self.use_lanczos = use_lanczos
        self.rank_lanczos = rank_lanczos        
        self.current_score = 0
        self.duration = 0
        self.state = 0      # 0 is normal, 1 is abnormal
        temp = np.random.rand(self.order[0])
        temp /= np.linalg.norm(temp)
        self.x = 0

    def fit(self, X, y=None):
        states = []
        count1 = 0
        for i in X:
            print("ccccccccccccccccccccccc count 1 is ",count1)
            print("ooooooooooooooooooooooo Order is ",self.order)
            print("ttttttttttttttttttttttt Threshold is ",self.threshold)
            self.predict_proba(i, count1, y)
            # Check to see if score is above threshold, if so, anomally has occured
            if self.current_score >= self.threshold:
                self.state=1 
            else:
                self.state=0
            states.append(self.state)
            count1 = count1+1
        return states  # returns array of either 0 or 1 / normal or abnormal
    
    def predict(self, X, y=None):
        states = []
        count2 = 0
        for j in X:
            print("ccccccccccccccccccccccc count 2 is ",count2)
            self.predict_proba(j, count2, y)
            # Check to see if score is above threshold, if so, anomally has occured
            if self.current_score >= self.threshold:
                self.state=1 
            else:
                self.state=0
            states.append(self.state)
            count2=count2+1
        return states  # returns array of either 0 or 1 / normal or abnormal
    
    def predict_proba(self, X, count, y=None):
        if count == 0:
            self.current_score, self.x = start_SST(startdata=X,win_length=self.win_length,
                n_component=self.n_components,order=self.order,lag=self.lag)
        else:
            self.current_score, self.x = stream_SST(stream=X,win_length=self.win_length,x0=self.x,
                n_component=self.n_components,order=self.order,lag=self.lag)
        return self.current_score # returns the score
#"""