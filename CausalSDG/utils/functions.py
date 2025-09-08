from typing import List, Union, Dict

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from numpy import ndarray
from pandas import DataFrame
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV

from .cpt import CPTData


def two_sample_test(original_dataset: DataFrame, synthetic_dataset: DataFrame,
                    features: List[str]):
    n_sample = min(len(original_dataset), len(synthetic_dataset))

    original = original_dataset[features].sample(n_sample)
    synthetic = synthetic_dataset[features].sample(n_sample)

    dataset = pd.concat(
        (original, synthetic), axis=0
    ).astype("category")

    X_dt, y_dt = dataset, np.asarray([0] * len(original) + [1] * len(synthetic))

    X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=0.2)

    cls = LGBMClassifier(verbose=-1)

    param_grid = {
        "num_leaves": [10, 20, 30],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [10, 30, 100],
        "max_depth": [-1, 5, 10],
        "objective": ["binary"],
        "n_jobs": [-1],
        "force_col_wise": [True],
    }

    grid_search = GridSearchCV(
        cls, param_grid, cv=5, n_jobs=-1, verbose=0, refit=True, scoring="f1")
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_

    y_predicted_train = model.predict(X_train)
    y_predicted_test = model.predict(X_test)
    return {
        "Accuracy train": metrics.accuracy_score(y_predicted_train, y_train),
        "Accuracy test": metrics.accuracy_score(y_predicted_test, y_test),

        "F1 score train": metrics.f1_score(y_predicted_train, y_train),
        "F1 score test": metrics.f1_score(y_predicted_test, y_test),

        "AUC train": metrics.roc_auc_score(y_predicted_train, y_train),
        "AUC test": metrics.roc_auc_score(y_predicted_test, y_test),
    }


def make_dist_from_dataframe2(data: DataFrame,
                              X: Union[List[str], str],
                              Y: str,
                              marginal_on: str = None) -> CPTData:
    """
    Constructs a Conditional Probability Table (CPT) from a pandas DataFrame where the child
    variable `Y` may contain lists (e.g., multiple labels/tags per instance).

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset containing both parent (`X`) and child (`Y`) variables.

    X : Union[List[str], str]
        One or more column names to use as parent (conditioning) variables.

    Y : str
        Column name representing the dependent variable. Must contain a list-like column.

    Returns:
    --------
    CPTData
        A Conditional Probability Table object representing P(Y | X), where for each combination
        of parent values, a probability distribution over the values of `Y` is stored.

    Example:
    --------
    df = pd.DataFrame({
         "parent": ["a", "a", "b"],
         "child": [["cat", "dog"], ["cat"], ["horse"]]
    })
    make_dist_from_dataframe(df, X="parent", Y="child")
    """

    # Ensure X is a list of column names
    if isinstance(X, str):
        X = [X]

    X = sorted(X)

    if isinstance(marginal_on, str):
        marginal_on = X.index(marginal_on)

    features = X + [Y]  # X is a list Y is string, so [Y] is a list

    # Explode the list in the Y column so each element gets its own row
    df = data[features].explode(Y)

    # Count the number of occurrences for each (X, Y) combination
    cross_dt = pd.crosstab(index=[df[i] for i in X], columns=df[Y])

    # Extract the list of all Y categories
    categories = list(cross_dt.columns)

    to_tuple = lambda x: (x,) if not isinstance(x, tuple) else x
    to_dist = lambda x: d if (d := np.asarray(x)).sum() == 0 else d / d.sum()

    prob_table = cross_dt.apply(
        axis=1,
        func=lambda x: pd.Series(data=[to_tuple(x.name), to_dist(x.values)],
                                 index=["occ", "dist"])
    ).set_index("occ")["dist"].to_dict()

    return CPTData(cpt=prob_table,
                   y_categories=categories,
                   marginal_on=marginal_on)


def exponential_tilting(dist: ndarray, lambda_: float) -> ndarray:
    """
    Applies exponential tilting to a given distribution. Exponential tilting is a technique
    where a distribution's probabilities are adjusted using the exponential function, influenced
    by a tilting parameter (lambda_). This operation modifies the weight of probabilities across
    the distribution to emphasize or de-emphasize specific elements.

    Parameters:
    -----------
    dist: ndarray
        A probability distribution represented as a sequence of values.
        Each value corresponds to the probability of a discrete event.

    lambda_: float
        The tilting parameter used to adjust the distribution. A non-zero value modifies
        the weights exponentially.

    Returns:
    --------
    ndarray
        The adjusted probability distribution after applying exponential tilting. The resulting
        probabilities are normalized to ensure their sum equals 1.
    """
    dis_0 = np.asarray([dist[k] * np.exp(-lambda_ * k) for k in range(len(dist))])
    dis_0 /= np.sum(dis_0)
    return dis_0


import math


def move_dist_mass(dist: Dict, alpha: float):
    """
    Modify a probability distribution based on an alpha parameter.

    With alpha = 0, the distribution remains unchanged.
    With alpha > 0, the distribution becomes more skewed to the left,
    With alpha < 0, the distribution becomes skewed to the right,

    Parameters:
    ----------
        dist (dict): Original distribution dictionary
        alpha (float): Unbalancing parameter.

    Returns:
    ----------
        dict: A new dictionary with the modified distribution.

    """
    if not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be an integer or a float.")

    categories = list(dist.keys())
    initial_prob = list(dist.values())

    if not categories:
        return {}

    num_categories = len(initial_prob)

    if alpha == 0:
        return dict(dist)

    unbalanced_dist = []
    for i in range(num_categories):
        peso = math.exp(-alpha * i)
        unbalanced_dist.append(initial_prob[i] * peso)

    sum_unbalanced_dist = sum(unbalanced_dist)

    if sum_unbalanced_dist == 0:
        first_not_null_index = -1
        for i in range(num_categories):
            if initial_prob[i] > 0:
                first_not_null_index = i
                break

        if first_not_null_index != -1:

            new_dist = [0.0] * num_categories
            new_dist[first_not_null_index] = 1.0
        else:
            new_dist = [0.0] * num_categories
    else:
        new_dist = [p / sum_unbalanced_dist for p in unbalanced_dist]

    return dict(zip(categories, new_dist))
