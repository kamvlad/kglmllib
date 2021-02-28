import numpy as np


class AddNaNIndicator:
    def __init__(self, columns, suffix='_is_NaN', copy=True):
        self.columns = columns
        self.suffix = suffix
        self.copy = copy

    def transform(self, df):
        if self.copy:
            df = df.copy()
        indicators = df[self.columns].isnull().astype(np.float16)
        for col in indicators.columns:
            df.insert(len(df.columns), col + self.suffix, indicators[col])
        return df
