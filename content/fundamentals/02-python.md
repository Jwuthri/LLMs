# 2. Python for machine learning

## Core idea

Python is the language that holds a machine learning project together.
You write a few lines of Python, and libraries such as NumPy do the heavy calculations in fast compiled code.
Almost all projects have the same four stages: load the data, clean it, fit a model, measure the error.

## The example

The [toy example](../toy-example.md) has no table of data, so this page uses its own small dataset.
Each row is one request to a language model server. All the values are assumed.

| Row | `tokens` (output tokens) | `model` | `seconds` (response time) |
|---|---|---|---|
| 0 | 100 | small | 1.2 |
| 1 | 200 | small | 2.1 |
| 2 | (missing) | large | 3.9 |
| 3 | 400 | large | 5.2 |
| 4 | 300 | small | 3.0 |
| 5 | 500 | large | 6.4 |

The goal is to predict `seconds` from `tokens` and `model`.
The input columns are the **features**. The column to predict is the **target**.

## 1. Python basics

### Setup

The data comes from a text file. Each value is text, also the numbers. The cell `tokens` of row 2 is empty.

### Steps

1. **Put each row in a dict and the table in a list.** A **list** keeps items in order. A **dict** gives a name to each value.

    ```python
    rows = [
        {"tokens": "100", "model": "small", "seconds": "1.2"},
        {"tokens": "200", "model": "small", "seconds": "2.1"},
        {"tokens": "", "model": "large", "seconds": "3.9"},
        # ... 3 more rows
    ]
    rows[0]["tokens"]    # '100': a string (str), not a number
    len(rows)            # 6
    ```

    **Why:** these two types hold most data before it goes into a grid of numbers (part 2). The type is important: `"100" + "200"` gives `"100200"`, not 300.

2. **Convert the text with a function, and handle the error.** `float("")` stops the program with a `ValueError`. The function catches this error for an empty cell only.

    ```python
    def to_number(text):
        try:
            return float(text)
        except ValueError:
            if text == "":
                return None      # an empty cell is a missing value
            raise                # other bad text must stop the program

    to_number("100")     # 100.0
    to_number("")        # None
    ```

    **Why:** real data always has bad cells. `try` / `except` lets you decide what to do for the case that you expect. For a case that you do not expect, a stop is better than a wrong number that nobody sees.

3. **Apply the function to each row.** A **list comprehension** is a loop that builds a list in one line.

    ```python
    tokens = [to_number(row["tokens"]) for row in rows]   # [100.0, 200.0, None, 400.0, 300.0, 500.0]
    known = [t for t in tokens if t is not None]          # 5 values
    sum(known) / len(known)                               # 1500 / 5 = 300.0
    ```

    **Why:** you do the same small operation on many items. A comprehension says "for each item, do this" without index variables.

4. **Put the values and their functions in a class.** A **class** is a template for objects that hold data and the functions that use it. This one learns two numbers (`fit`), then uses them (`transform`): the mean, and the **standard deviation** (how far the values are from the mean).

    ```python
    class Standardizer:
        def fit(self, values):                # np is the NumPy library (part 2)
            self.mean = np.mean(values)
            self.std = np.std(values)
            return self

        def transform(self, values):
            return (np.asarray(values) - self.mean) / self.std

    scaler = Standardizer().fit(known)    # mean 300.0, std 141.421
    scaler.transform([100.0])             # [-1.414]
    ```

    **Why:** the mean and the standard deviation must stay with the code that uses them. This style is object-oriented programming. scikit-learn and PyTorch use the same pattern, so you will read many classes.

<div class="key-point">

Four tools cover most machine learning scripts: lists and dicts for the data, functions for each operation, `try` / `except` for bad input, and classes for objects that learn something and keep it.

</div>

### From toy to real

- A real project puts the exact version of each library in a lock file. Then a different person gets the same numbers.
- Type hints such as `def to_number(text: str) -> float | None` tell the reader and the tools what a function accepts.
- Plain Python loops are slow. They are good for 6 rows, not for 6 million. Part 2 removes them.

## 2. Data science libraries

### Setup

**NumPy** gives Python the **array**: a grid of numbers that all have the same type.
We put each column in one array. A missing number becomes `nan` ("not a number").

### Steps

1. **Make the arrays.**

    ```python
    tokens = np.array([100, 200, np.nan, 400, 300, 500])    # shape (6,), dtype float64
    seconds = np.array([1.2, 2.1, 3.9, 5.2, 3.0, 6.4])
    model = np.array(["small", "small", "large", "large", "small", "large"])
    ```

    **Why:** an array keeps its numbers in one block of memory, and the loops run in compiled code. The **shape** `(6,)` says: one axis, 6 items.

2. **Calculate on a full column at one time.** This is **vectorization**. The time for 100 tokens is:

    ```python
    seconds / tokens * 100    # [1.2, 1.05, nan, 1.3, 1.0, 1.28]
    ```

    **Why:** one line replaces a loop, and it reads like the formula. Row 2 stays `nan`, so you can see that its result is unknown.

3. **Calculate a summary, and take care with `nan`.** `np.mean(tokens)` gives `nan`. `np.nanmean(tokens)` ignores the missing value and gives 300.0.

    **Why:** almost each calculation with `nan` gives `nan`. This is a safety function: you cannot forget a missing value.

4. **Select rows with a mask.** `model == "small"` gives an array of `True` and `False`. Use it as an index.

    ```python
    seconds[model == "small"].mean()    # (1.2 + 2.1 + 3.0) / 3 = 2.1
    seconds[model == "large"].mean()    # (3.9 + 5.2 + 6.4) / 3 = 5.167
    ```

    **Why:** "the mean of one group" is the most common question about a table. The large model is slower, so `model` is a useful feature.

5. **Measure the relation between two columns, then draw it.** The **correlation** is a number from −1 to 1 that says if two columns move together. On the 5 rows with a known `tokens`, it is 0.985 for `tokens` and `seconds`. **Matplotlib** draws the figure of part 4.

    **Why:** look at the data before you fit a model. A value near 1 says that a straight line is a good first model.

Real-library sketch (the tests do not run this code). **pandas** puts the columns in one table with names, a `DataFrame`:

```python
import pandas as pd

df = pd.DataFrame(rows)
df["tokens"] = pd.to_numeric(df["tokens"], errors="coerce")   # text that is not a number becomes NaN
df["seconds"] = df["seconds"].astype(float)
df["tokens"].isna().sum()                  # the number of missing values
df.groupby("model")["seconds"].mean()      # the mean of each group, as in step 4
df.plot.scatter(x="tokens", y="seconds")   # pandas calls Matplotlib
```

<div class="key-point">

Think in columns, not in rows. One operation on an array replaces a loop, and it is much faster on large data.

</div>

### From toy to real

- A `float64` number uses 8 bytes. A table of 1,000,000 rows and 2 columns uses 16,000,000 bytes (16 MB).
- **Seaborn** is a library on top of Matplotlib. It makes statistical plots in one line.
- A PyTorch **tensor** is an array with two more functions: it can be on a graphics processing unit (GPU), and it can calculate gradients. See [Neural networks](03-neural-networks.md).

## 3. Data preprocessing

### Setup

A model needs a matrix of numbers with no holes, and with columns of a similar size.
Our table has three problems: one missing value, one text column, and one column with large numbers (hundreds) next to a column of 0 and 1.
One rule controls this part: each number that you learn from the data must come from the train rows only.

<figure class="diagram" data-src="python/pipeline.svg">
<figcaption>The train rows give the numbers. The test rows only receive them.</figcaption>
</figure>

### Steps

1. **Split the rows first.** Shuffle the row numbers with a **seed** (the start number of the random generator): `np.random.default_rng(5).permutation(6)`. The first 4 are the **train set** (rows 1, 2, 3, 4). The last 2 are the **test set** (rows 0, 5).

    **Why:** the test rows play the role of new data. The shuffle is necessary because files are frequently sorted. The seed makes the split the same on each run.

2. **Fill the missing value.** The train rows have the `tokens` values 200, `nan`, 400, 300. Their **median** (the middle value) is 300. Row 2 gets 300. This operation is **imputation**.

    **Why:** a model cannot calculate with `nan`. If you remove the row, you lose 25% of the train set. The median is safer than the mean when a column has some extreme values.

3. **Put the numbers on the same scale.** **Standardization** subtracts the mean and divides by the standard deviation. The train rows give mean = 300 and standard deviation = 70.711.

    $$z = \frac{x - \text{mean}}{\text{std}} \qquad \text{row 1: } \frac{200 - 300}{70.711} = -1.414$$

    | Row | 0 (test) | 1 | 2 | 3 | 4 | 5 (test) |
    |---|---|---|---|---|---|---|
    | `tokens` | 100 | 200 | 300 (filled) | 400 | 300 | 500 |
    | $z$ | −2.828 | −1.414 | 0 | 1.414 | 0 | 2.828 |

    A second method is **min-max scaling**, $(x - \min) / (\max - \min)$, with the train values 200 and 400. Row 1 gives 0, row 3 gives 1, and the test row 5 gives 1.5.

    **Why:** many models work badly when one column has much larger numbers than the others, because that column controls the result. The test rows fall outside the train range (−2.828 and 2.828). That is correct: the model must show what it does with values that it did not see.

4. **Change the text column into numbers.** **One-hot encoding** makes one column for each category: `small` = [1, 0] and `large` = [0, 1]. We keep only the second column, `is_large` = [0, 0, 1, 1, 0, 1].

    **Why:** you cannot multiply the word "large" by a weight. We drop one column because the two columns always add up to 1. With a column of ones in the model (part 4), there would be 4 columns but only 3 independent ones, and the equations would have no single solution.

<div class="key-point">

Fit on the train set, then apply to all the rows. If a number from the test rows gets into the preparation, the test result is too good. This error is **data leakage**.

</div>

### From toy to real

- Real projects use three sets: train, **validation** (to compare settings) and test (used one time, at the end).
- An **outlier** is a value far from the others. A common rule flags $|z| > 3$. Look at each one before you remove it: it can be an error or a rare, real event.
- For text, the preparation step is the tokenizer. See [The LLM architecture](../scientist/01-architecture.md).

## 4. Machine learning libraries

### Setup

**Linear regression** predicts the target as a weighted sum of the features: $\hat{y} = \theta_0 + \theta_1 z + \theta_2 \cdot \text{is\_large}$.
The best weights $\theta$ give the smallest mean squared error on the train rows.
For this model there is a direct formula, the **normal equation**: $\theta = (X^\top X)^{-1} X^\top y$.

### Steps

1. **Build the matrix $X$ and the vector $y$ from the 4 train rows.** The columns of $X$ are: 1, $z$, `is_large`.

    $$X = \begin{bmatrix} 1 & -1.414 & 0 \\ 1 & 0 & 1 \\ 1 & 1.414 & 1 \\ 1 & 0 & 0 \end{bmatrix} \qquad y = \begin{bmatrix} 2.1 \\ 3.9 \\ 5.2 \\ 3.0 \end{bmatrix}$$

    **Why:** the column of ones gives the model a base value, $\theta_0$, that does not depend on the features.

2. **Calculate $X^\top X$ and $X^\top y$.** Each entry is a dot product of two columns. For example, the first entry of $X^\top y$ is 2.1 + 3.9 + 5.2 + 3.0 = 14.2.

    $$X^\top X = \begin{bmatrix} 4 & 0 & 2 \\ 0 & 4 & 1.414 \\ 2 & 1.414 & 2 \end{bmatrix} \qquad X^\top y = \begin{bmatrix} 14.2 \\ 4.384 \\ 9.1 \end{bmatrix}$$

    **Why:** the equations $X^\top X \theta = X^\top y$ say "the gradient of the squared error is 0". That is the minimum. See [Mathematics](01-math.md).

3. **Solve the 3 equations.** `np.linalg.solve` gives $\theta = [3.1,\ 0.778,\ 0.9]$. Check the first equation: 4 · 3.1 + 0 · 0.778 + 2 · 0.9 = 14.2.

    **Why:** you can read a linear model. A request to the small model with 300 tokens takes 3.1 s. One more standard deviation of tokens (70.711) adds 0.778 s, which is 0.011 s for each token. The large model adds 0.9 s.

4. **Predict the test rows.**

    | Row | Calculation | Prediction | Real value |
    |---|---|---|---|
    | 0 | 3.1 + 0.778 · (−2.828) + 0.9 · 0 | 0.9 | 1.2 |
    | 5 | 3.1 + 0.778 · 2.828 + 0.9 · 1 | 6.2 | 6.4 |

    **Why:** the model did not see these rows. They show how the model works on new data.

5. **Measure the error, and compare with simpler models.** The **mean squared error (MSE)** on the test rows is ((0.9 − 1.2)² + (6.2 − 6.4)²) / 2 = 0.065. Its square root is 0.255 s.

    | Model | Train MSE | Test MSE |
    |---|---|---|
    | Always answer the train mean (3.55 s) | 1.3125 | 6.8225 |
    | `tokens` only | 0.111 | 0.3125 |
    | `tokens` and `is_large` | 0.010 | 0.065 |

    **Why:** an error has a meaning only next to a **baseline** (a very simple model). And the train error (0.010) is usually too optimistic: the weights were fitted to those rows.

<figure class="plot">
<img src="/figures/python/regression.svg" alt="Response time against output tokens. Two parallel model lines, one for the small model and one 0.9 s higher for the large model. The 4 train rows are near the lines. The 2 test rows, at 100 and 500 tokens, are 0.3 s and 0.2 s above their lines.">
<figcaption>The model is two parallel lines. The test rows are outside the range of the train rows, and the errors are larger there.</figcaption>
</figure>

Real-library sketch (the tests do not run this code). **scikit-learn** has a class for each step of parts 3 and 4, and a `Pipeline` that fits them on the train rows only:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train, test = df.iloc[[1, 2, 3, 4]], df.iloc[[0, 5]]   # train_test_split(df, test_size=2) makes a random split
prepare = ColumnTransformer([
    ("tokens", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["tokens"]),
    ("model", OneHotEncoder(categories=[["small", "large"]], drop="first"), ["model"]),
])
pipeline = Pipeline([("prepare", prepare), ("regression", LinearRegression())])
pipeline.fit(train[["tokens", "model"]], train["seconds"])
pipeline.predict(test[["tokens", "model"]])
```

<div class="key-point">

A model in scikit-learn is an object with `fit` (learn from the train rows) and `predict` (answer for new rows). Always measure it on rows that it did not see, and next to a baseline.

</div>

### From toy to real

Each of these classes has a `fit` method. The classes of the first five rows also have `predict`. `PCA` and `TSNE` have `fit_transform`, which returns the new columns.

| Goal | Algorithm | How it works | scikit-learn class |
|---|---|---|---|
| Predict a number | Linear regression | Weighted sum of the features | `LinearRegression` |
| Predict a category | Logistic regression | Weighted sum, then a function that gives a probability | `LogisticRegression` |
| Number or category | Decision tree, random forest | A chain of yes/no questions. A forest is many trees that vote | `DecisionTreeClassifier`, `RandomForestClassifier` |
| Number or category | k-nearest neighbors (K-NN) | The answer of the k most similar train rows | `KNeighborsClassifier` |
| Find groups, with no target | K-means | Move k centres to the middle of their points | `KMeans` |
| Reduce the number of columns | PCA, t-SNE | PCA keeps the directions with the largest eigenvalues. t-SNE makes a 2-D map for plots | `PCA`, `TSNE` |

A large language model follows the same stages. But its output is not a weighted sum of the features, so there is no direct formula for its billions of weights: it uses gradient descent.

## Minimal code

This code gives the same numbers as parts 3 and 4.

```python
import numpy as np

tokens = np.array([100, 200, np.nan, 400, 300, 500])
is_large = np.array([0., 0., 1., 1., 0., 1.])
seconds = np.array([1.2, 2.1, 3.9, 5.2, 3.0, 6.4])

order = np.random.default_rng(5).permutation(6)
train, test = np.sort(order[:4]), np.sort(order[4:])           # [1 2 3 4], [0 5]

tokens = np.where(np.isnan(tokens), np.nanmedian(tokens[train]), tokens)   # fill with 300
z = (tokens - tokens[train].mean()) / tokens[train].std()      # mean 300, std 70.711

X = np.column_stack([np.ones(6), z, is_large])
theta = np.linalg.solve(X[train].T @ X[train], X[train].T @ seconds[train])   # [3.1, 0.778, 0.9]
predicted = X[test] @ theta                                    # [0.9, 6.2]
mse = np.mean((predicted - seconds[test]) ** 2)                # 0.065
```

## Comparison

| Stage | NumPy (this page) | Library tool | Result in the example |
|---|---|---|---|
| Load | List of dicts, `np.array` | `pandas.DataFrame` | 6 rows, 1 missing value |
| Split | `rng.permutation` | `train_test_split` | Train 1, 2, 3, 4. Test 0, 5 |
| Fill | `np.nanmedian` | `SimpleImputer` | 300 |
| Scale | `(x - mean) / std` | `StandardScaler` | Mean 300, std 70.711 |
| Encode | `model == "large"` | `OneHotEncoder` | `is_large` |
| Fit | `np.linalg.solve` | `LinearRegression` | θ = [3.1, 0.778, 0.9] |
| Measure | `np.mean((a - b) ** 2)` | `mean_squared_error` | Test MSE 0.065 |

## Checks and common misconceptions

- **"Scale all the data, then split."** This is a leak. In part 1 the class used all the known values and got a standard deviation of 141.421. The train rows alone give 70.711. Check: each `fit` in your code must get train rows only.
- **"The test error is the true error of the model."** With 6 rows, it is mostly luck. For the seeds 0 to 9, the test MSE of the same model goes from 0.001 to 2.256. Check: use more data, or **cross-validation** (do the split many times and take the mean of the errors).
- **"All libraries calculate the same standard deviation."** `np.std` divides by $n$ and gives 70.711. The `.std()` of pandas divides by $n - 1$ and gives 81.650 for the same 4 values. `StandardScaler` divides by $n$.

## Sources

This page follows section 2 of "LLM Fundamentals" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [Real Python](https://realpython.com/), articles and tutorials.
- [Learn Python](https://www.youtube.com/watch?v=rfscVS0vtbw), freeCodeCamp.
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/), Jake VanderPlas.
- [Machine Learning for Everybody](https://youtu.be/i_LwzRVP7bg), freeCodeCamp.
- [Intro to Machine Learning](https://www.udacity.com/course/intro-to-machine-learning--ud120), Udacity.
- [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), scikit-learn documentation (it divides by $n$).
- [DataFrame.std](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.std.html), pandas documentation (it divides by $n - 1$).
