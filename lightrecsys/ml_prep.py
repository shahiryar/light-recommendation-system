import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
)


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


class CategoricalEncoder:
    """
    A class used to encode categorical variables in a dataset.

    ...

    Attributes
    ----------
    method : str
        the method to encode categorical variables ('label_encoding', 'one_hot_encoding')
    encoder : OneHotEncoder or LabelEncoder
        an encoder object from sklearn.preprocessing

    Methods
    -------
    fit(X)
        Fits the encoder on X.
    transform(X)
        Transforms X.
    """

    def __init__(self, method="one_hot", prefix=None):
        """
        Constructs all the necessary attributes for the CategoricalEncoder object.

        Parameters
        ----------
            method : str, optional
                the method to encode categorical variables ('label', 'one_hot') (default is 'one_hot')
        """
        self.method = method
        self.prefix = prefix
        if method == "one_hot":
            self.encoder = None
        else:
            self.encoder = LabelEncoder()

    def fit(self, X):
        """
        Fits the encoder on X.

        Parameters
        ----------
            X : pandas DataFrame
                the dataset to fit the encoder on
        """
        if self.encoder is None:
            return
        self.encoder.fit(X)

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
        if self.method == "one_hot":
            if self.prefix is None:
                return pd.get_dummies(X, dtype=int)
            return pd.get_dummies(X, prefix=self.prefix, dtype=int)
        else:
            return pd.DataFrame(self.encoder.transform(X), columns=X.columns)


class Scaler:
    """
    A class used to scale numerical variables in a dataset.

    ...

    Attributes
    ----------
    scaler : StandardScaler or MinMaxScaler
        a scaler object from sklearn.preprocessing

    Methods
    -------
    fit(X)
        Fits the scaler on X.
    transform(X)
        Transforms X.
    """

    def __init__(self, method="standard"):
        """
        Constructs all the necessary attributes for the Scaler object.

        Parameters
        ----------
            method : str, optional
                the method to scale numerical variables ('standard', 'minmax') (default is 'standard')
        """
        if method == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

    def fit(self, X):
        """
        Fits the scaler on X.

        Parameters
        ----------
            X : pandas DataFrame
                the dataset to fit the scaler on
        """
        self.scaler.fit(X)

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
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns)
