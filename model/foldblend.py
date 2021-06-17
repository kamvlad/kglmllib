import pathlib
import numpy as np
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm


class FoldBlend:
    def __init__(self,
                 modelfactory,
                 folds=None,
                 random_state=None):
        self.folds = folds
        self.modelfactory = modelfactory
        self.random_state = random_state

    def fit(self, X, y, eval_data=None, mlflow_log=True):
        folder = KFold(n_splits=self.folds, shuffle=True, random_state=self.random_state)
        self.models = []
        for fold_idx, indices in tqdm(enumerate(folder.split(X, y)), total=folder.n_splits):
            train_idx, _ = indices
            train_fold_features = X.iloc[train_idx]
            train_fold_target = y.iloc[train_idx]
            model = self.modelfactory()
            model.fit(train_fold_features, train_fold_target)
            self.models.append(model)

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        predictions = np.vstack(predictions)
        return np.mean(predictions, axis=0)

    def save(self, path: pathlib.Path):
        pass

    def load(self, path: pathlib.Path):
        pass