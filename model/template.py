import pathlib


class Regressor:
    def __init__(self):
        pass

    def fit(self, X, y, eval_data=None, mlflow_log=True):
        pass

    def predict(self, X):
        pass

    def save(self, path: pathlib.Path, name=None):
        pass

    @staticmethod
    def load(path: pathlib.Path, name=None):
        pass