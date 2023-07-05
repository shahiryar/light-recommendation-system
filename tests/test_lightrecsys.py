#!/usr/bin/env python

"""Tests for `lightrecsys` package."""

import pytest
import unittest
from click.testing import CliRunner

import pandas as pd
import numpy as np
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


def test_categorical_encoder():
    # Create a simple dataframe for testing
    df = pd.DataFrame(
        {
            "fruit": ["apple", "banana", "cherry", "apple", "banana", "cherry"],
            "color": ["red", "yellow", "red", "green", "yellow", "red"],
        }
    )

    # Test one-hot encoding
    one_hot_encoder = CategoricalEncoder(method="one_hot")
    one_hot_encoder.fit(df)
    transformed_df = one_hot_encoder.transform(df)

    assert transformed_df.shape == (
        6,
        5,
    )  # Check the shape of the transformed dataframe
    assert set(transformed_df.columns) == set(
        ["fruit_apple", "fruit_banana", "fruit_cherry", "color_red", "color_yellow"]
    )  # Check the column names

    # Test label encoding
    label_encoder = CategoricalEncoder(method="label_encoding")
    label_encoder.fit(df)
    transformed_df = label_encoder.transform(df)

    assert transformed_df.shape == (
        6,
        2,
    )  # Check the shape of the transformed dataframe
    assert set(transformed_df.columns) == set(
        ["fruit", "color"]
    )  # Check the column names
    assert (
        transformed_df["fruit"].nunique() == 3
    )  # Check the number of unique values in the 'fruit' column
    assert (
        transformed_df["color"].nunique() == 2
    )  # Check the number of unique values in the 'color' column


def test_Scaler():
    # Create a simple dataframe
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [100, 200, 300, 400, 500]})

    # Test StandardScaler
    scaler_standard = Scaler(method="standard")
    df_scaled_standard = scaler_standard.fit(df).transform(df)
    assert np.isclose(df_scaled_standard.mean(), 0).all()
    assert np.isclose(df_scaled_standard.std(), 1).all()

    # Test MinMaxScaler
    scaler_minmax = Scaler(method="minmax")
    df_scaled_minmax = scaler_minmax.fit(df).transform(df)
    assert np.isclose(df_scaled_minmax.min(), 0).all()
    assert np.isclose(df_scaled_minmax.max(), 1).all()
