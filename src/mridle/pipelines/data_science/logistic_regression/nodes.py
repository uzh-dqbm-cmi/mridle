import pandas as pd


def remove_na(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Changes variables for model optimization modifying feature_df

    Args:
        dataframe: dataframe obtained from feature generation

    Returns: modified dataframe specific for this model
    """

    dataframe = dataframe.dropna(axis=0).reset_index(drop=True)

    return dataframe
