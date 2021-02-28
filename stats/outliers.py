import numpy as np
import scipy.stats


def iqr_outliers(df, k = 1.5):
    """For standard normal distribution has 0.7% outliers"""
    med = df.median(axis=0)
    q1 = df.quantile(0.25, axis=0)
    q3 = df.quantile(0.75, axis=0)
    iqr = q3 - q1
    return (df <= (q1 - k * iqr)) | (df >= (q3 + k * iqr))

