import pandas as pd


def process_features_for_model(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Changes variables for model optimization modifying feature_df

    Args:
        dataframe: dataframe obtained from feature generation

    Returns: modified dataframe specific for this model
    """

    dataframe = dataframe.dropna()

    return dataframe
