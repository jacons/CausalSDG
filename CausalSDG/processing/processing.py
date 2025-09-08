"""
================================================================================
Author: Andrea Iommi
Code Ownership:
    - All Python source code in this file is written solely by the author.
Documentation Notice:
    - All docstrings and inline documentation are written by ChatGPT,
      but thoroughly checked and approved by the author for accuracy.
================================================================================
"""

import abc
import warnings
from abc import ABC
from typing import Any, List, Dict

from pandas import Series


class AbstractFunction:
    """
    Function that has a "get" method and "feature_name" and "depends on" attribute.

    Parameters:
    -----------
        feature_name (str): The name of the feature.
    """

    def __init__(self, feature_name: str, apply_on: List[str]):
        self.feature_name = feature_name
        self.apply_on = apply_on

    @abc.abstractmethod
    def transform(self, *args, **kwargs) -> Any:
        pass


class ProcessingFunction(AbstractFunction, ABC):
    """
    A function applied after data generation to modify or transform features.

    Parameters:
    -----------
        feature_name (str): The name of the feature.
        depends_on (List[str]): A list of feature names (parents) that the feature depends on.
        replace_dependencies (bool): Whether to replace dependencies with the computed feature.
    """

    def __init__(self, feature_name: str, apply_on: List[str], replace: bool):
        super().__init__(feature_name, apply_on)

        if len(apply_on) > 1 and replace:
            warnings.warn("replace will be disable because the dependency are more than 1")
            self.replace = False
        else:
            self.replace = replace


class FunctionApplying:
    """
    Applies a sequence of functions to a dataset. The functions provided are typically instances
    of PostProcessingFunction.
    """

    def __init__(self, functions: Dict[str, ProcessingFunction] = None):

        if functions is None:
            functions = {}

        self.functions: Dict[str, ProcessingFunction] = functions

    def __call__(self, args: Series, exclude_fun: List[str]) -> Series:
        return self.transform(args=args, exclude_fun=exclude_fun)

    def transform(self, args: Series, exclude_fun: List[str] = None) -> Series:
        """
        Applies each function in sequence to the given dataset sample.

        Parameters:
        -----------
            args (Series): A Pandas Series representing a single data sample.

        Returns:
        -----------
            Series: A Pandas Series containing the computed features.
        """

        if exclude_fun is None:
            exclude_fun = []

        output = {}
        for name, sampler in self.functions.items():
            if name not in exclude_fun:
                apply_on = sampler.apply_on
                name = sampler.feature_name
                output[name] = sampler.transform(*args[apply_on].to_list())

        return Series(output)
