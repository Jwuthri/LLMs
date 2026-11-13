"""Numbers and plots for content/fundamentals/02-python.md.

Run: uv run python -m examples.fundamentals.python_ml
"""

import numpy as np

from examples.plotstyle import save_figure

PAGE = "python"

# The dataset of the page: 6 requests to a language model server (assumed values).
# Each value is text, as in a CSV file. The empty string is a missing value.
RAW_ROWS = [
    {"tokens": "100", "model": "small", "seconds": "1.2"},
    {"tokens": "200", "model": "small", "seconds": "2.1"},
    {"tokens": "", "model": "large", "seconds": "3.9"},
    {"tokens": "400", "model": "large", "seconds": "5.2"},
    {"tokens": "300", "model": "small", "seconds": "3.0"},
    {"tokens": "500", "model": "large", "seconds": "6.4"},
]

SEED = 5
TEST_ROWS = 2


# --- Python basics -------------------------------------------------------------------------

def to_number(text: str) -> float | None:
    """Convert the text of one cell to a number. An empty cell gives None."""
    try:
        return float(text)
    except ValueError:
        if text == "":
            return None
        raise


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


class Standardizer:
    """Learn the mean and the standard deviation of a column, then scale values with them."""

    def fit(self, values) -> "Standardizer":
        values = np.asarray(values, dtype=float)
        self.mean = float(values.mean())
        self.std = float(values.std())  # population standard deviation (divide by n)
        return self

    def transform(self, values) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.std


# --- Data science libraries: the columns as NumPy arrays -----------------------------------

TOKENS_LIST = [to_number(row["tokens"]) for row in RAW_ROWS]
TOKENS = np.array([np.nan if value is None else value for value in TOKENS_LIST])
SECONDS = np.array([float(row["seconds"]) for row in RAW_ROWS])
MODEL = np.array([row["model"] for row in RAW_ROWS])


def group_means() -> dict[str, float]:
    """Mean of `seconds` for each value of `model` (a boolean mask selects the rows)."""
    return {name: float(SECONDS[MODEL == name].mean()) for name in ("small", "large")}


def correlation_tokens_seconds() -> float:
    """Pearson correlation on the rows where `tokens` is known."""
    known = ~np.isnan(TOKENS)
    return float(np.corrcoef(TOKENS[known], SECONDS[known])[0, 1])


# --- Data preprocessing --------------------------------------------------------------------

def split(seed: int = SEED, test_rows: int = TEST_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle the row numbers with a seed. The last `test_rows` are the test set."""
    order = np.random.default_rng(seed).permutation(len(RAW_ROWS))
    return np.sort(order[:-test_rows]), np.sort(order[-test_rows:])


def one_hot(values: np.ndarray, categories: tuple[str, ...] = ("small", "large")) -> np.ndarray:
    """One column for each category: 1 in the column of the category of the row, 0 elsewhere."""
    return np.array([[float(value == category) for category in categories] for value in values])


def preprocess(seed: int = SEED) -> dict:
    """Split, fill the missing value, scale and encode. Each statistic comes from the train rows only."""
    train, test = split(seed)
    median = float(np.nanmedian(TOKENS[train]))
    filled = np.where(np.isnan(TOKENS), median, TOKENS)
    scaler = Standardizer().fit(filled[train])
    z = scaler.transform(filled)
    low, high = filled[train].min(), filled[train].max()
    encoded = one_hot(MODEL)
    return {
        "train": train, "test": test, "median": median, "filled": filled,
        "mean": scaler.mean, "std": scaler.std, "std_sample": float(filled[train].std(ddof=1)),
        "z": z, "minmax": (filled - low) / (high - low),
        "one_hot": encoded, "is_large": encoded[:, 1],
    }


# --- Machine learning: linear regression with the normal equation --------------------------

def design_matrix(prep: dict, with_model: bool = True) -> np.ndarray:
    """Columns: 1 (intercept), scaled tokens, and is_large if `with_model`."""
    columns = [np.ones(len(RAW_ROWS)), prep["z"]]
    if with_model:
        columns.append(prep["is_large"])
    return np.column_stack(columns)


def mse(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean((predicted - actual) ** 2))


def fit_regression(prep: dict, with_model: bool = True) -> dict:
    """Solve (X^T X) theta = X^T y on the train rows, then measure the error on both sets."""
    train, test = prep["train"], prep["test"]
    x = design_matrix(prep, with_model)
    xtx = x[train].T @ x[train]
    xty = x[train].T @ SECONDS[train]
    theta = np.linalg.solve(xtx, xty)
    predicted = x @ theta
    return {
        "X_train": x[train], "y_train": SECONDS[train], "XtX": xtx, "Xty": xty, "theta": theta,
        "predicted": predicted,
        "mse_train": mse(predicted[train], SECONDS[train]),
        "mse_test": mse(predicted[test], SECONDS[test]),
    }


def baseline_mse(prep: dict) -> dict[str, float]:
    """Train and test error of a model that always answers the mean of the train targets."""
    train, test = prep["train"], prep["test"]
    answer = SECONDS[train].mean()
    return {
        "train": mse(np.full(len(train), answer), SECONDS[train]),
        "test": mse(np.full(len(test), answer), SECONDS[test]),
    }


def collinear_rank(prep: dict) -> int:
    """Rank of X with the intercept and the two one-hot columns (4 columns)."""
    x = np.column_stack([np.ones(len(RAW_ROWS)), prep["z"], prep["one_hot"]])
    return int(np.linalg.matrix_rank(x[prep["train"]]))


def mse_test_by_seed(seeds: range = range(10)) -> list[float]:
    """Test error of the full model for other splits of the same 6 rows."""
    return [fit_regression(preprocess(seed))["mse_test"] for seed in seeds]


# --- Plot ----------------------------------------------------------------------------------

def make_figures() -> None:
    prep = preprocess()
    theta = fit_regression(prep)["theta"]
    train, test = prep["train"], prep["test"]

    def draw(ax, theme):
        xs = np.array([50.0, 550.0])
        for offset, style, name in ((0.0, "-", "small"), (theta[2], "--", "large")):
            ys = theta[0] + theta[1] * (xs - prep["mean"]) / prep["std"] + offset
            ax.plot(xs, ys, style, color=theme["series"][2], linewidth=1.5, label=f"model line, {name}")
        ax.scatter(prep["filled"][train], SECONDS[train], s=46, color=theme["series"][0], zorder=3,
                   label="train rows")
        ax.scatter(prep["filled"][test], SECONDS[test], s=46, marker="s", color=theme["series"][1], zorder=3,
                   label="test rows")
        for row in range(len(RAW_ROWS)):
            note = f"{row}: {MODEL[row]}" + (" (filled)" if np.isnan(TOKENS[row]) else "")
            ax.annotate(note, (prep["filled"][row], SECONDS[row]), textcoords="offset points",
                        xytext=(8, -10), fontsize=8, color=theme["ink2"])
        ax.set_xlabel("Output tokens")
        ax.set_ylabel("Response time (seconds)")
        ax.set_xlim(50, 620)
        ax.legend(loc="upper left", fontsize=8)

    save_figure(draw, PAGE, "regression")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    print("--- Python basics")
    print("first row:", RAW_ROWS[0], " rows:", len(RAW_ROWS))
    print("to_number('100'), to_number(''):", to_number("100"), to_number(""))
    print("tokens:", TOKENS_LIST)
    known = [value for value in TOKENS_LIST if value is not None]
    print("known:", known, "sum:", sum(known), "mean:", mean(known))
    scaler = Standardizer().fit(known)
    print("Standardizer on the 5 known values: mean", scaler.mean, "std", scaler.std,
          "transform(100):", scaler.transform([100.0]))

    print("--- data science libraries")
    print("tokens array:", TOKENS, TOKENS.shape, TOKENS.dtype)
    print("seconds per 100 tokens:", SECONDS / TOKENS * 100)
    print("np.mean:", np.mean(TOKENS), " np.nanmean:", np.nanmean(TOKENS))
    print("mean seconds by model:", group_means())
    print("correlation tokens, seconds:", correlation_tokens_seconds())

    print("--- data preprocessing")
    prep = preprocess()
    for key in ("train", "test", "median", "filled", "mean", "std", "std_sample", "z", "minmax", "one_hot"):
        print(f"{key}:", prep[key])
    print("rank of X with intercept + 2 one-hot columns:", collinear_rank(prep), "(4 columns)")

    print("--- linear regression")
    full = fit_regression(prep)
    for key in ("X_train", "y_train", "XtX", "Xty", "theta", "predicted"):
        print(f"{key}:", full[key], sep="\n")
    print("XtX theta:", full["XtX"] @ full["theta"])
    print("seconds for each token:", full["theta"][1] / prep["std"])
    print("train residuals:", (full["predicted"] - SECONDS)[prep["train"]])
    print("test residuals:", (full["predicted"] - SECONDS)[prep["test"]])
    print("mse train:", full["mse_train"], " mse test:", full["mse_test"], " rmse test:", np.sqrt(full["mse_test"]))
    one = fit_regression(prep, with_model=False)
    print("tokens only: theta", one["theta"], "mse train", one["mse_train"], "mse test", one["mse_test"])
    print("baseline: train mean", SECONDS[prep["train"]].mean(), "mse", baseline_mse(prep))
    print("test mse for seeds 0-9:", np.array(mse_test_by_seed()))
    make_figures()
