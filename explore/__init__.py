def columns_with_nan(df):
    return df.columns[df.isnull().any()]
