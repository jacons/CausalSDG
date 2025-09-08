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

from typing import List, Union

from numpy import ndarray
from numpy.random import choice

from .processing import ProcessingFunction


class ShrinkFeatureProcessing(ProcessingFunction):
    """
    A processing function that replaces occurrences of a specified "shrink" category in a feature
    with a randomly sampled value from a given set of categories based on a probability distribution.

    This is useful in scenarios where a particular category (e.g., a placeholder like "other" or "unknown")
    should be collapsed into more informative or likely categories.

    Attributes:
    -----------
    categories : List[str]
        List of possible replacement categories.

    p : ndarray
        A probability distribution corresponding to the replacement `categories`.
        Must sum to 1 and match the length of `categories`.

    shrink : List[str]
        List of category values that should be replaced by a sampled alternative.
        If a single string is passed, it will be converted to a list.
    """

    def __init__(self, feature_name: str, categories: List[str], p: ndarray,
                 shrink: Union[str, List[str]]):

        super().__init__(feature_name="_" + feature_name, apply_on=[feature_name], replace=True)

        self.categories = categories
        self.p = p
        self.shrink = [shrink] if isinstance(shrink, str) else shrink

    def transform(self, x: Union[str, None]) -> Union[str, None]:
        """
        Transforms the feature value by replacing the "shrink" category with a sampled value.

        Parameters:
        -----------
        x : str or None
            The original feature value.

        Returns:
        --------
        str or None
            A randomly sampled category if `x` equals `shrink`; otherwise, returns the original value.
        """
        if x in self.shrink:
            return str(choice(self.categories, p=self.p, size=1)[0])
        return x
