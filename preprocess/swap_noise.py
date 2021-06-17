import numpy as np


def swap_noise(features_dict, probs, inplace=True):
    if inplace:
        result = features_dict
    else:
        result = {}
    mask = {}

    for feature in features_dict:
        p = probs.get(feature, 0.0)
        values = features_dict[feature]
        if p > 0.0:
            mask[feature] = np.random.binomial(1, p, size=len(values))
            shuffle_idx = np.random.choice(len(values), replace=True)
            idx = np.where(mask[feature] == 0, np.arange(len(values)), shuffle_idx)
            result[feature] = values[idx]
        else:
            result[feature] = values
            mask[feature] = np.zeros(len(values))
    return result, mask
