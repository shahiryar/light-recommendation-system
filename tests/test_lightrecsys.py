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
    # Create a sample dataset with missing values
    data = {
        "A": [1, 2, None, 4, 5],
        "B": [6, None, 8, 9, 10],
        "C": [11, 12, 13, None, 15],
    }
    df = pd.DataFrame(data)

    # Create an instance of MissingValueImputer
    imputer = MissingValueImputer(method="mean")

    # Test fit method
    imputer.fit(df)

    # Test transform method
    transformed_df = imputer.transform(df)

    # Check if missing values are imputed
    assert transformed_df.isnull().sum().sum() == 0

    # Check if the transformed dataframe has the same shape
    assert transformed_df.shape == df.shape

    # Check if column names are preserved
    assert transformed_df.columns.tolist() == df.columns.tolist()

    # Check if the imputed values are correct
    expected_mean_A = df["A"].mean()
    expected_mean_B = df["B"].mean()
    expected_mean_C = df["C"].mean()

    assert transformed_df["A"].equals(pd.Series([1, 2, expected_mean_A, 4, 5]))
    assert transformed_df["B"].equals(pd.Series([6, expected_mean_B, 8, 9, 10]))
    assert transformed_df["C"].equals(pd.Series([11, 12, 13, expected_mean_C, 15]))

    print("Imputer: All tests passed successfully.")


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
