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

import ast
from typing import List, Union, Any

import numpy as np

from .processing import ProcessingFunction


class YearExpProcessing(ProcessingFunction):
    """
    Processes experience years by converting ranges into a random integer within the specified range.
    """

    @staticmethod
    def transform(x: str) -> int:

        if x == "Not specified":
            return 0
        elif x == "+15 years":
            start, end = 15, 30
        else:
            x = x.split()
            start, end = int(x[0]), int(x[2])

        return int(np.random.randint(start, end + 1, 1)[0])


class AgeProcessing(ProcessingFunction):
    """
    Processes age ranges by converting them into a random integer within the specified range.
    """

    def transform(self, x: str) -> Union[str, int]:

        if x == "No answer":
            return "No answer"
        if x == "65+":
            start, end = 65, 75
        else:
            x = x.split("-")
            start, end = int(x[0]), int(x[1])

        return int(np.random.randint(start, end + 1, 1)[0])


class Str2ListProcessing(ProcessingFunction):
    """
    Converts a string representation of a list into an actual list of strings.
    """

    def __init__(self, feature_name: str, apply_on: List[str], replace: bool,
                 take_first: bool = False):
        super().__init__(feature_name, apply_on, replace)

        self.take_first = take_first

    def transform(self, x: str) -> List[str]:

        if isinstance(x, list):
            to_list = x
        elif isinstance(x, str):
            to_list = ast.literal_eval(x)
        else:
            raise ValueError("Input must be of type str")

        return (to_list[0] if len(to_list) > 0 else None) if self.take_first else to_list


class ProcessingCompose(ProcessingFunction):

    def __init__(self, compose: List[ProcessingFunction]):
        """
        Initializes the ProcessingCompose with a list of ProcessingFunction objects.
        """
        feature_name = compose[0].feature_name
        apply_on = compose[0].apply_on
        replace = compose[0].replace

        if not (all(obj.feature_name == feature_name for obj in compose) and
                all(obj.apply_on == apply_on for obj in compose) and
                all(obj.replace == replace for obj in compose)):
            raise ValueError(
                "All functions in compose must have the same feature_name, apply_on, and replace attributes.")

        self.compose = compose
        super().__init__(feature_name, apply_on, replace)

    def transform(self, x: str) -> Any:

        for func in self.compose:
            x = func.transform(x)

        return x


class RandomProficiency(ProcessingFunction):
    def __init__(self, feature_name: str, apply_on: List[str], replace: bool):
        super().__init__(feature_name, apply_on, replace)

    def transform(self, x: List[str]) -> dict[str, float]:
        return {i: np.random.choice([0.25, 0.50, 0.75, 1]) for i in x}
