import pathlib

import numpy as np
import pandas as pd
import mlflow

from tqdm.notebook import tqdm

from sklearn.model_selection import KFold


def regression_cv(features,
                  labels,
                  model_factory,
                  metric,
                  folds,
                  mlflow_tags,
                  artifacts_dir,
                  random_state):
    train_results = {}
    test_results = {}

    train_folds_errors = []
    test_folds_errors = []

    with mlflow.start_run(tags=({'stage': 'evaluation', **mlflow_tags})) as evaluation_run:
        mlflow.log_params({
            'n_splits': folds,
            'cv_random_state': random_state
        })

        # create splitter
        if random_state is not None:
            folder = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        else:
            folder = KFold(n_splits=folds, shuffle=False)

        for fold_idx, indices in tqdm(enumerate(folder.split(features, labels)), total=folder.n_splits):
            with mlflow.start_run(nested=True, tags={'fold': fold_idx}) as fold_run:
                train_idx, test_idx = indices

                train_fold_features = features.iloc[train_idx]
                train_fold_target = labels.iloc[train_idx]

                test_fold_features = features.iloc[test_idx]
                test_fold_target = labels.iloc[test_idx]

                # train
                model = model_factory()
                model.fit(train_fold_features, train_fold_target,
                          eval_data=(test_fold_features, test_fold_target),
                          mlflow_log=True)
                fold_model_path = pathlib.Path(artifacts_dir) / fold_run.info.run_id / 'evaluation_models' / f'fold_{fold_idx}'
                fold_model_path.mkdir(parents=True, exist_ok=True)
                model.save(fold_model_path)

                # predict
                train_pred = model.predict(train_fold_features)
                test_pred = model.predict(test_fold_features)

                # fold error
                fold_train_error = metric(train_fold_target, train_pred)
                fold_test_error = metric(test_fold_target, test_pred)
                train_folds_errors.append(fold_train_error)
                test_folds_errors.append(fold_test_error)

                # store prediction
                append_array(train_results, 'fold', [fold_idx] * len(train_idx))
                append_array(train_results, 'sample_idx', train_idx)
                append_array(train_results, 'pred', train_pred)
                append_array(train_results, 'true', train_fold_target)
                append_array(train_results, 'fold_error', [fold_train_error] * len(train_idx))

                append_array(test_results, 'fold', [fold_idx] * len(test_idx))
                append_array(test_results, 'sample_idx', test_idx)
                append_array(test_results, 'pred', test_pred)
                append_array(test_results, 'true', test_fold_target)
                append_array(test_results, 'fold_error', [fold_test_error] * len(test_idx))

                print(f'Fold #{fold_idx}: test = {fold_test_error:0.5f}, train = {fold_train_error:0.5f}')

                # mlflow log
                mlflow.log_metrics({
                    'test_err': fold_test_error,
                    'train_err': fold_train_error
                })

        folds_stat_df = pd.DataFrame({'fold': range(len(train_folds_errors)),
                                      'train': train_folds_errors,
                                      'test': test_folds_errors})
        train_results_df = pd.DataFrame(train_results)
        test_results_df = pd.DataFrame(test_results)
        print(f'Test error: {folds_stat_df["test"].mean():0.5f} ({folds_stat_df["test"].std():0.5f}),' +
              f'Train error: {folds_stat_df["train"].mean():0.5f} ({folds_stat_df["train"].std():0.5f})')

        # mlflow log
        mlflow.log_metrics({
            'test_err': folds_stat_df["test"].mean(),
            'test_err_std': folds_stat_df["test"].std(),
            'train_err': folds_stat_df["train"].mean(),
            'train_err_std': folds_stat_df["train"].std()
        })
        artifacts_path = pathlib.Path(artifacts_dir) / evaluation_run.info.run_id
        artifacts_path.mkdir(parents=True, exist_ok=True)
        train_results_df.to_csv(artifacts_path / 'evaluation_train.csv.zip', index=False)
        test_results_df.to_csv(artifacts_path / 'evaluation_test.csv.zip', index=False)
        mlflow.log_artifacts(artifacts_dir)

    return train_results_df, test_results_df, folds_stat_df


def append_array(dictionary, key, array):
    if key not in dictionary:
        dictionary[key] = array
    else:
        dictionary[key] = np.concatenate((dictionary[key], array))
