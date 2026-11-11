"""Each number shown in docs/fundamentals/02-python.md."""

import numpy as np
import pytest

from examples.fundamentals.python_ml import (
    MODEL, RAW_ROWS, SECONDS, TOKENS, TOKENS_LIST, Standardizer,
    baseline_mse, collinear_rank, correlation_tokens_seconds, fit_regression, group_means,
    mean, mse_test_by_seed, one_hot, preprocess, split, to_number,
)


def test_python_basics():
    assert len(RAW_ROWS) == 6 and RAW_ROWS[0]["tokens"] == "100"
    assert to_number("100") == 100.0 and to_number("") is None
    with pytest.raises(ValueError):
        float("")
    with pytest.raises(ValueError):
        to_number("fast")
    assert TOKENS_LIST == [100.0, 200.0, None, 400.0, 300.0, 500.0]
    known = [value for value in TOKENS_LIST if value is not None]
    assert sum(known) == 1500.0 and mean(known) == 300.0


def test_standardizer_class():
    scaler = Standardizer().fit([100.0, 200.0, 400.0, 300.0, 500.0])
    assert scaler.mean == 300.0
    assert round(scaler.std, 3) == 141.421
    assert np.allclose(scaler.transform([100.0, 300.0]), [-1.414, 0.0], atol=5e-4)


def test_numpy_columns():
    assert TOKENS.shape == (6,) and TOKENS.dtype == np.float64
    assert TOKENS.itemsize * 1_000_000 * 2 == 16_000_000  # bytes for 1,000,000 rows and 2 columns
    assert np.allclose(SECONDS / TOKENS * 100, [1.2, 1.05, np.nan, 1.3, 1.0, 1.28], equal_nan=True)
    assert np.isnan(np.mean(TOKENS)) and np.nanmean(TOKENS) == 300.0
    means = group_means()
    assert round(means["small"], 3) == 2.1 and round(means["large"], 3) == 5.167
    assert round(correlation_tokens_seconds(), 3) == 0.985


def test_split():
    train, test = split()
    assert train.tolist() == [1, 2, 3, 4] and test.tolist() == [0, 5]


def test_preprocessing():
    prep = preprocess()
    assert prep["median"] == 300.0
    assert prep["filled"].tolist() == [100, 200, 300, 400, 300, 500]
    assert prep["mean"] == 300.0
    assert round(prep["std"], 3) == 70.711
    assert round(prep["std_sample"], 3) == 81.650
    assert np.allclose(prep["z"], [-2.828, -1.414, 0, 1.414, 0, 2.828], atol=5e-4)
    assert np.allclose(prep["minmax"], [-0.5, 0, 0.5, 1, 0.5, 1.5])


def test_one_hot():
    assert one_hot(MODEL).tolist() == [[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]]
    assert preprocess()["is_large"].tolist() == [0, 0, 1, 1, 0, 1]
    assert collinear_rank(preprocess()) == 3  # 4 columns, but only 3 independent ones


def test_normal_equation():
    result = fit_regression(preprocess())
    assert np.allclose(result["X_train"], [[1, -1.414, 0], [1, 0, 1], [1, 1.414, 1], [1, 0, 0]], atol=5e-4)
    assert np.allclose(result["y_train"], [2.1, 3.9, 5.2, 3.0])
    assert np.allclose(result["XtX"], [[4, 0, 2], [0, 4, 1.414], [2, 1.414, 2]], atol=5e-4)
    assert np.allclose(result["Xty"], [14.2, 4.384, 9.1], atol=5e-4)
    assert np.allclose(result["theta"], [3.1, 0.778, 0.9], atol=5e-4)
    assert np.isclose(result["theta"][1] / preprocess()["std"], 0.011)


def test_predictions_and_errors():
    prep = preprocess()
    result = fit_regression(prep)
    assert np.allclose(result["predicted"], [0.9, 2.0, 4.0, 5.1, 3.1, 6.2])
    assert np.allclose((result["predicted"] - SECONDS)[prep["test"]], [-0.3, -0.2])
    assert round(result["mse_train"], 3) == 0.01
    assert round(result["mse_test"], 3) == 0.065
    assert round(float(np.sqrt(result["mse_test"])), 3) == 0.255


def test_simpler_models_are_worse():
    prep = preprocess()
    tokens_only = fit_regression(prep, with_model=False)
    assert np.allclose(tokens_only["theta"], [3.55, 1.096], atol=5e-4)
    assert np.isclose(tokens_only["mse_test"], 0.3125)
    assert round(float(SECONDS[prep["train"]].mean()), 3) == 3.55
    assert round(tokens_only["mse_train"], 3) == 0.111
    assert np.isclose(baseline_mse(prep)["train"], 1.3125)
    assert np.isclose(baseline_mse(prep)["test"], 6.8225)


def test_error_depends_on_the_split():
    errors = mse_test_by_seed()
    assert round(min(errors), 3) == 0.001 and round(max(errors), 3) == 2.256
