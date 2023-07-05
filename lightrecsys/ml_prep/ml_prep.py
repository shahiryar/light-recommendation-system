import pandas as pd
from sklearn.impute import SimpleImputer


class MissingValueImputer:
    """
    A class used to impute missing values in a dataset.

    ...

    Attributes
    ----------
    imputer : SimpleImputer
        a SimpleImputer object from sklearn.impute

    Methods
    -------
    fit(X)
        Fits the imputer on X.
    transform(X)
        Transforms X.
    """

    def __init__(self, method="mean", fill_value=None):
        """
        Constructs all the necessary attributes for the MissingValueImputer object.

        Parameters
        ----------
            method : str, optional
                the method to handle missing values ('drop', 'mean', 'median', 'mode', 'constant') (default is 'mean')
            fill_value : any, optional
                if method='constant', the value to fill missing values with (default is None)
        """
        if method == "constant":
            strategy = "constant"
            fill_value = fill_value
        else:
            strategy = method
            fill_value = None
        self.imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)

    def fit(self, X):
        """
        Fits the imputer on X.

        Parameters
        ----------
            X : pandas DataFrame
                the dataset to fit the imputer on
        """
        self.imputer.fit(X)

    def transform(self, X):
        """
        Transforms X.

        Parameters
        ----------
            X : pandas DataFrame
                the dataset to transform

        Returns
        -------
        pandas DataFrame
            the transformed dataset
        """
        return pd.DataFrame(self.imputer.transform(X), columns=X.columns)


# Similar docstrings can be added to the CategoricalEncoder and Scaler classes.
