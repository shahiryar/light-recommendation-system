#!/usr/bin/env python

"""Tests for `lightrecsys` package."""

import pytest
import unittest
from click.testing import CliRunner

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lightrecsys.ml_prep import MissingValueImputer, CategoricalEncoder, Scaler

from lightrecsys import cli


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "lightrecsys.cli.main" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output


def test_missing_value_imputer():
    # Create a DataFrame with some missing values
    data = pd.DataFrame(
        {
            "A": [1, 2, np.nan, 4, 5],
            "B": [np.nan, 2, 3, 4, 5],
            "C": [1, 2, 3, np.nan, np.nan],
        }
    )

    # Test mean imputation
    imputer = MissingValueImputer(method="mean")
    imputer.fit(data)
    result = imputer.transform(data)
    assert not result.isnull().any().any()  # There should be no missing values
    assert (
        result.loc[2, "A"] == 3
    )  # The missing value in column A should be replaced with the mean (3)

    # Test median imputation
    imputer = MissingValueImputer(method="median")
    imputer.fit(data)
    result = imputer.transform(data)
    assert not result.isnull().any().any()  # There should be no missing values
    assert (
        result.loc[0, "B"] == 3
    )  # The missing value in column B should be replaced with the median (3)

    # Test constant imputation
    imputer = MissingValueImputer(method="constant", fill_value=0)
    imputer.fit(data)
    result = imputer.transform(data)
    assert not result.isnull().any().any()  # There should be no missing values
    assert (
        result.loc[3, "C"] == 0
    )  # The missing value in column C should be replaced with 0


from sklearn.preprocessing import LabelEncoder
import numpy as np


def test_categorical_encoder():
    # Create a sample dataset
    data = pd.DataFrame({"Category": ["A", "B", "A", "C", "B"]})

    # Test one-hot encoding without prefix
    encoder = CategoricalEncoder(method="one_hot")
    transformed = encoder.transform(data)
    expected = pd.DataFrame(
        {
            "Category_A": [1, 0, 1, 0, 0],
            "Category_B": [0, 1, 0, 0, 1],
            "Category_C": [0, 0, 0, 1, 0],
        }
    )
    assert transformed.equals(expected), "One-hot encoding without prefix test failed"

    # Test one-hot encoding with prefix
    encoder = CategoricalEncoder(method="one_hot", prefix="Cat")
    transformed = encoder.transform(data)
    expected = pd.DataFrame(
        {"Cat_A": [1, 0, 1, 0, 0], "Cat_B": [0, 1, 0, 0, 1], "Cat_C": [0, 0, 0, 1, 0]}
    )
    assert transformed.equals(expected), "One-hot encoding with prefix test failed"

    # Test label encoding
    encoder = CategoricalEncoder(method="label")
    encoder.fit(data)
    transformed = encoder.transform(data)
    expected = pd.DataFrame({"Category": [0, 1, 0, 2, 1]})
    assert transformed.equals(expected), "Label encoding test failed"

    print("Categorical Encoding: All tests passed successfully!")


def test_scaler():
    # Load the Iris dataset for testing
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Test StandardScaler
    scaler = Scaler(method="standard")
    scaler.fit(X)
    transformed_X = scaler.transform(X)

    # Verify the transformed dataset
    sklearn_scaler = StandardScaler()
    sklearn_scaler.fit(X)
    expected_transformed_X = pd.DataFrame(
        sklearn_scaler.transform(X), columns=X.columns
    )
    assert transformed_X.equals(expected_transformed_X), "StandardScaler test failed!"

    # Test MinMaxScaler
    scaler = Scaler(method="minmax")
    scaler.fit(X)
    transformed_X = scaler.transform(X)

    # Verify the transformed dataset
    sklearn_scaler = MinMaxScaler()
    sklearn_scaler.fit(X)
    expected_transformed_X = pd.DataFrame(
        sklearn_scaler.transform(X), columns=X.columns
    )
    assert transformed_X.equals(expected_transformed_X), "MinMaxScaler test failed!"

    print("Scaler: All tests passed successfully!")
