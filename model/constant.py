import pathlib

import numpy as np


class ConstantRegressor:
    def fit(self, X, y, eval_data=None, mlflow_log=True):
        self.mean = y.mean()

    def predict(self, X):
        return np.ones(X.shape[0]) * self.mean

    def save(self, path: pathlib.Path, name=None):
        pass

    @staticmethod
    def load(path: pathlib.Path, name=None):
        pass