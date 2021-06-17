import pathlib
import mlflow

from category_encoders import target_encoder

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge

# Model

class RidgeRegression:
    def __init__(self,
                 cat_features=None,
                 scaler=None,
                 alpha=0.1,
                 random_state=None):
        self.params = {
            'cat_features': cat_features,
            'scaler': scaler,
            'alpha': alpha,
            'random_state': random_state
        }

    def fit(self, X, y, eval_data=None, sample_weight=None):
        # TODO Treat nans
        if self.params['cat_features'] is not None:
            self.cat_encoder = target_encoder.TargetEncoder(cols=self.params['cat_features'])
            X = self.cat_encoder.fit_transform(X, y)

        if self.params['scaler'] is not None:
            if self.params['scaler'] == 'StandardScaler':
                self.scaler = StandardScaler()
            elif self.params['scaler'] == 'MinMaxScaler':
                self.scaler = MinMaxScaler()
            else:
                assert 'Unknown scaler' # may be scaler (check is fitted)
            X = self.scaler.fit_transform(X)

        self.model = Ridge(self.params['alpha'])
        self.model.fit(X, y, sample_weight)

        if mlflow.active_run():
            pass #
        train_report = {}
        return train_report

    def predict(self, X):
        # TODO Treat nans
        if self.cat_encoder is not None:
            X = self.cat_encoder.transform(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def save(self, path: pathlib.Path):
        self.params = 0

    def load(self, path: pathlib.Path, params_only: bool):
        self.params = 0 # depickle
        if not params_only:
            self.cat_encoder = 0 # depicle
            self.scaler = 0
            self.model = 0

    def set_params(self, **params):
        pass

    def copy(self, params_only: bool):
        pass


# Analyze


# Tuning

def tuning(X, y, base):
    pass