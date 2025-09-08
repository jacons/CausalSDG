import warnings
from typing import List, Dict, Tuple, Union, Literal

import numpy as np
import pandas as pd
from numpy import ndarray, zeros
from pandas import DataFrame


def custom_warning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'


warnings.formatwarning = custom_warning


class PTData:
    """
    A class representing a Probability Table (PT) for categorical variables.

    Attributes:
    -----------
    pt : Dict[Tuple[str, ...], np.ndarray]
        probability vector

    y_categories : List[str|int|float]
        List of categories for the dependent variable Y.

    is_valid : bool
        Indicates whether the provided PT structure is valid (values conform to expected formats).

    Example:
    --------
        pt_data = PTData(
            cpt={
                (,): np.array([0.4, 0.6]),
            },
            y_categories=["horse", "cat"]
        )
    """

    def __init__(self, cpt: Dict[Tuple, Union[ndarray, List[float]]], y_categories: List[Union[str, int, float]]):

        self.cpt = cpt
        for k, v in cpt.items():
            if isinstance(v, list):
                self.cpt[k] = np.array(v)

        self.y_categories = y_categories
        self._dim_input = None

        self.is_valid = self.check_distribution_validity()
        self._dim_output = len(y_categories)

    def __repr__(self):
        """
        Returns a human-readable string representation of the CPT.
        """
        return f"CPTData(cpt={self.cpt}, y_categories={self.y_categories})"

    def __str__(self):
        output = "Conditional Probability Table:\n"
        output += f"  Dependent Variable Categories: {self.y_categories}\n"
        output += "  Probabilities:\n"
        for conditions, probabilities in self.cpt.items():
            output += f"    P(Y | {', '.join(conditions)}) = {probabilities}\n"
        return output

    def __len__(self):
        """
        Returns the number of parent configurations in the CPT.
        """
        return len(self.cpt)

    def copy(self) -> 'PTData':
        return PTData(cpt=self.cpt.copy(), y_categories=self.y_categories.copy())

    def check_distribution_validity(self) -> bool:
        """
        Validates the structure and consistency of the CPT.

        Returns:
        --------
        bool
            True if the CPT is valid, False otherwise

        Prints:
        -------
        Error messages are printed if invalid formats or distributions are detected.
        """

        if len(self.cpt) != 1:
            print(f"CPT must have only one key, but found {len(self.cpt)} keys.")
            return False

        if self.cpt.get(tuple()) is None:
            print("CPT must have a key with an empty tuple.")
            return False
        if not isinstance(self.cpt[tuple()], ndarray):
            print("CPT value must be a numpy array.")
            return False
        if len(self.cpt[tuple()]) != len(self.y_categories):
            print("CPT value length must match the number of categories.")
            return False
        if not np.isclose(np.sum(self.cpt[tuple()]), 1.0):
            print("CPT value must sum to 1.0.")
            return False

        self._dim_input = len(list(self.cpt.keys())[0])
        return True

    def get_probability(self, X: Tuple[str, ...], Y: str) -> float:
        """
        Returns the probability of Y given parent configuration X.

        Parameters:
        -----------
        X : Tuple[str, ...]
            A specific parent configuration.

        Y : str
            The category of the dependent variable.

        Returns:
        --------
        float
            The conditional probability P(Y | X).

        Raises:
        -------
        KeyError:
            If `X` is not in the CPT or `Y` is not in the category list.
        """
        if X not in self.cpt:
            raise KeyError(f"Conditions '{X}' not found in the CPT.")
        if Y not in self.y_categories:
            raise KeyError(f"Category '{Y}' not found in y_categories.")

        category_index = self.y_categories.index(Y)
        return float(self.cpt[X][category_index])

    def get_distribution(self, X: Tuple[str, ...]) -> ndarray:
        # Returns the probability distribution over the variable Y,
        return self.cpt[X]

    def get_dim_input(self) -> int:
        # Returns the number of parent variables (dimensions) in the CPT.
        return self._dim_input

    def get_dim_output(self) -> int:
        # Returns the number of categories (dimensions) for the dependent variable Y.
        return self._dim_output

    def to_dataframe(self) -> DataFrame:
        # Converts the CPT to a pandas DataFrame for easier visualization and manipulation.
        dt = (pd.DataFrame(self.cpt)
        .T
        .rename(columns={i: v for i, v in enumerate(self.y_categories)})
        .sort_index()[sorted(self.y_categories)]
        )
        return dt

    def sample(self, X: Tuple[str, ...], elements: int = 1) -> List[Union[str, int, float]]:
        """
        Samples elements from the conditional probability distribution given a parent configuration.

        Parameters:
        -----------
        X : Tuple[str, ...]
            A tuple representing a specific configuration of parent variables

        elements : int, default=1
            The number of elements to sample from the distribution.
        """

        if X is None:
            raise ValueError("X cannot be None")
        if elements < 1:
            raise ValueError("elements must be greater than 0")

        dist = self.get_distribution(X=X)
        value = np.random.choice(self.y_categories, p=dist, size=elements, replace=False)
        return value.tolist()


class CPTData(PTData):
    """
    A class representing a Conditional Probability Table (CPT) for categorical variables.

    Attributes:
    -----------
    cpt : Dict[Tuple[str, ...], np.ndarray]
        Dictionary mapping parent configurations (as tuples of strings) to probability vectors
        over the dependent variable's categories.

    y_categories : List[str|int|float]
        List of categories for the dependent variable Y.

    is_valid : bool
        Indicates whether the provided CPT structure is valid (keys and values conform to expected formats).

    idx_feature : Optional[int]
        If specified, indicates which feature (index in the condition tuple) was marginalized.

    marginalized_cpt : Optional[CPTData]
        A secondary CPTData object holding the marginalized table if `marginal_on` was used.

    Example:
    --------
        cpt_data = CPTData(
            cpt={
                ("a",): np.array([0.4, 0.6]),
                ("b",): np.array([0, 1.0])
            },
            y_categories=["horse", "cat"]
        )
    """

    def __init__(self, cpt: Dict[Tuple[str, ...], Union[ndarray, List[float]]],
                 y_categories: List[Union[str,int,float]],
                 marginal_on: int = None):
        """
        Initializes the Conditional Probability Table (CPT).

        Parameters:
        -----------
        cpt : Dict[Tuple[str, ...], np.ndarray]
            A mapping from parent configurations (tuples of strings) to probability vectors (ndarray).
            Each vector must have the same length as `y_categories` and sum to 1 (or be all zeros).

        y_categories : List[str]
            A list of categories for the dependent variable Y.
        """
        super().__init__(cpt=cpt, y_categories=y_categories)

        if isinstance(marginal_on, int) and marginal_on >= 0:
            self.idx_feature = marginal_on
            self.marginalized_cpt = self.marginal_on(idx_feature=marginal_on)
        else:
            self.idx_feature = None
            self.marginalized_cpt = None

    def marginal_on(self, idx_feature: int) -> 'CPTData':
        """
        Marginalizes the CPT with respect to the feature at the given index in the parent configuration tuple.

        Parameters:
        -----------
        idx_feature : int
            Index of the feature in the parent configuration tuple to marginalize over.

        Returns:
        --------
        CPTData
            A new CPTData object representing the marginalized distribution P(Y | parent[idx_feature]).
        """
        tmp = {}

        for k, v in self.cpt.items():
            if k[idx_feature] not in tmp:
                tmp[k[idx_feature]] = v.copy()
            else:
                tmp[k[idx_feature]] += v

        # normalize to 1 -> build a probability distribution
        tmp = {(k,): v / v.sum() for k, v in tmp.items()}
        return CPTData(cpt=tmp, y_categories=self.y_categories.copy())

    def copy(self) -> 'CPTData':
        return CPTData(cpt=self.cpt.copy(), y_categories=self.y_categories.copy())

    def get_distribution(self,
                         X: Tuple[str, ...],
                         fill_missing: Literal["zero", "uniform"] = "zero",
                         verbose: bool = False) -> ndarray:
        """
        Returns the conditional probability distribution over the dependent variable Y,
        given a configuration of parent variables X.

        Parameters:
        -----------
        X : Tuple[str, ...]
            A tuple representing a specific configuration of parent variables

        fill_missing : Literal["zero", "uniform"], default="zero"
            Specifies the strategy to use when the provided configuration `X`
            is not found in the CPT or maps to a zero vector:

            - "zero": returns a vector of zeros.
            - "uniform": returns a uniform distribution over the categories of Y.

            If the CPT was initialized with a marginalization index (`marginal_on`),
            and the value at that index exists in the marginalized CPT, that distribution
            will be used instead of the fallback.

        verbose : bool, default=False
            If True, emits a warning when the configuration `X` is not found
            and a fallback distribution is returned.

        Returns:
        --------
        ndarray
            A 1D NumPy array representing the probability distribution over the categories of Y.
            The ordering of values corresponds to the `y_categories` attribute.

        Examples:
        ---------
        cpt = {
            ("sunny",): np.array([0.7, 0.3]),
             ("rainy",): np.array([0.2, 0.8])
        }
        cpt_data = CPTData(cpt, y_categories=["go_out", "stay_in"])
        cpt_data.get_distribution(("sunny",))
        array([0.7, 0.3])

        cpt_data.get_distribution(("cloudy",), fill_missing="uniform")
        array([0.5, 0.5])

        Raises:
        -------
        None

        Notes:
        ------
        This method is robust to missing keys and zero-sum vectors in the CPT.
        If a marginalization index is provided, it attempts to return a fallback
        marginal distribution before using the specified `fill_missing` strategy.
        """
        if X not in self.cpt or self.cpt[X].sum() == 0:

            if self.idx_feature is not None and self.marginalized_cpt is not None:
                if verbose:
                    warnings.warn(message=f"{X} not found in the CPT, returning marginalized distribution on "
                                          f"'{X[self.idx_feature]}'.")

                return self.marginalized_cpt.get_distribution(X=(X[self.idx_feature],),
                                                              fill_missing=fill_missing,
                                                              verbose=verbose)

            if verbose:
                warnings.warn(message=f"{X} not found in the CPT, returning '{fill_missing}' vector.")

            length = len(self.y_categories)
            dist = np.full(length, 1.0 / length) if fill_missing == "uniform" else zeros(length)

        else:
            dist = self.cpt[X]

        return dist

    def check_distribution_validity(self, verbose: bool = False) -> bool:
        """
        Validates the structure and consistency of the CPT.

        Returns:
        --------
        bool
            True if the CPT is valid, False otherwise

        Prints:
        -------
        Error messages are printed if invalid formats or distributions are detected.

        A valid CPT satisfies:
        - All keys are tuples of strings.
        - Each value is a numpy array of probabilities.
        - Arrays are of the same length as `y_categories`.
        - Each array sums to 1.0 or is entirely zero (to represent missing/no data).
        """
        zero_dist, only_one_dist = 0, 0

        for k, v in self.cpt.items():
            # Validate key type: must be a tuple of strings
            if not isinstance(k, tuple) or not all(isinstance(ki, str) for ki in k):
                print(f"All keys must be tuples of strings. {k}")
                return False

            # Validate value type: must be (list, np.ndarray) pair with proper dimensions
            if not isinstance(v, ndarray):
                print(f"All arrays must be numpy arrays. Key : {k}")
                return False

            if len(v) != len(self.y_categories):
                print(f"All arrays must have the same length. Key :{k}")
                return False

            if not (np.isclose(np.sum(v), 1.0) or np.isclose(np.sum(v), 0)):
                print(f"Invalid distribution format or probabilities do not sum to 1. Key : {k}")
                return False
            zero_dist += 1 if np.isclose(np.sum(v), 0) else 0
            only_one_dist += 1 if np.count_nonzero(v) == 1 else 0
        if verbose:
            print("Zero distribution :", zero_dist, "on", len(self.cpt.items()))
            print("Only one element distribution", only_one_dist, "on", len(self.cpt.items()))

        self._dim_input = len(list(self.cpt.keys())[0])
        return True

    def sample(self,
               X: tuple[str, ...],
               elements: int = 1,
               fill_missing: Literal["zero", "uniform"] = "zero",
               verbose: bool = False) -> List[Union[str, int, float]]:
        """
        Samples elements from the conditional probability distribution given a parent configuration.

        Parameters:
        -----------
        X : Tuple[str, ...]
            A tuple representing a specific configuration of parent variables

        elements : int, default=1
            The number of elements to sample from the distribution.

        fill_missing : Literal["zero", "uniform"], default="zero"
            Specifies the strategy to use when the provided configuration `X`
            is not found in the CPT or maps to a zero vector:

            - "zero": returns a vector of zeros.
            - "uniform": returns a uniform distribution over the categories of Y.

            If the CPT was initialized with a marginalization index (`marginal_on`),
            and the value at that index exists in the marginalized CPT, that distribution
            will be used instead of the fallback.

        verbose : bool, default=False
            If True, emits a warning when the configuration `X` is not found
            and a fallback distribution is returned.
        """

        if X is None:
            raise ValueError("X cannot be None")
        if elements < 1:
            raise ValueError("elements must be greater than 0")

        dist = self.get_distribution(X=X, fill_missing=fill_missing, verbose=verbose)
        value = np.random.choice(self.y_categories, p=dist, size=elements, replace=False)
        return value.tolist()
