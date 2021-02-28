import time
import pickle
import pathlib

import mlflow

import catboost as cgb
from .. import utils


class CatboostRegressor:
    def __init__(self, cat_features, random_state):
        self.cgb_params = {'loss_function': 'RMSE', 'random_seed': random_state}
        self.cat_features = cat_features

    def fit(self, X, y, eval_data=None, mlflow_log=True):
        start_time = time.time()
        train_pool = cgb.Pool(X, y, cat_features=self.cat_features)
        self.model = cgb.train(train_pool, self.cgb_params, verbose=False, plot=False)
        duration = time.time() - start_time

        if mlflow_log:
            self.log()
            mlflow.log_metric('train_duration', duration)
            mlflow.log_text('train_duration_str', utils.duration_str(duration))

    def log(self):
        mlflow.log_params(self.model.get_params())
        for pool, metrics in self.model.get_evals_result().items():
            for metric_name, values in metrics.items():
                for step, value in enumerate(values):
                    mlflow.log_metric(f'{pool}__{metric_name}', value, step)

    def predict(self, X):
        test_pool = cgb.Pool(X, cat_features=self.cat_features)
        return self.model.predict(test_pool)

    def save(self, path: pathlib.Path, name=None):
        if name is None:
            name = 'catboost_model'
        self.model.save_model(str(path / name))
        with open(path / (name + '_params.pckl'), 'wb') as fd:
            pickle.dump({'cat_features': self.cat_features}, fd)

    @staticmethod
    def load(path: pathlib.Path, name=None):
        if name is None:
            name = 'catboost_model'
        model = cgb.CatBoost()
        model.load_model(str(path / name))
        with open(path / (name + '_params.pckl'), 'rb') as fd:
            params = pickle.load(fd)
            cat_features = params['cat_features']

        reg = CatboostRegressor(cat_features, model.get_param('random_seed'))
        reg.model = model
        return reg