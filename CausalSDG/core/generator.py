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

import json
import warnings
from pathlib import Path
from typing import Tuple, Union, List, Dict, Literal, Callable, Hashable

import pandas as pd
from dowhy import gcm
from dowhy.gcm import StructuralCausalModel
from dowhy.gcm.auto import AssignmentQuality, assign_causal_mechanisms
from dowhy.gcm.causal_mechanisms import StochasticModel, FunctionalCausalModel
from dowhy.gcm.falsify import falsify_graph
from networkx.classes import DiGraph
from pandas import DataFrame

from CausalSDG.processing import FunctionApplying
from .causal_mechanisms import ConditionalMechanismFromDict

Mechanism = Union[StochasticModel, FunctionalCausalModel]


def custom_warning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'


warnings.formatwarning = custom_warning

class CausalGenerator:
    """
    A synthetic data generator that uses a structural causal model (SCM) for generating data
    optionally applies post-processing transformations and supports model evaluation and falsification.

    Attributes:
    -----------
    causal_model : gcm.StructuralCausalModel
        The structural causal model representing the underlying data-generating process.

    direct_graph : DiGraph
        A directed acyclic graph (DAG) that encodes causal dependencies between variables.

    pre_processing : FunctionApplying
        A container of transformation functions applied to the training dataset before
        fitting the causal model. Useful for cleaning, imputing, or augmenting real data
        prior to modeling.

    post_processing : FunctionApplying
        A container of transformation functions applied after synthetic data is generated.
        These transformations may add derived features, apply formatting, or replace variables.
    """

    def __init__(self, direct_graph: DiGraph, pre_processing: FunctionApplying = None,
                 post_processing: FunctionApplying = None):
        """
        Initializes the generator with a causal model and optional post-processing function.

        Parameters:
        -----------
        direct_graph : DiGraph
            The directed acyclic graph representing causal relationships between variables.

        pre_processing : FunctionApplying, optional
            A collection of transformation functions to apply to the training dataset before
            model fitting. Defaults to an empty set of functions.

        post_processing : FunctionApplying, optional
            A collection of transformation functions to apply to each generated sample
            after data generation. Defaults to an empty set of functions.
        """
        self.causal_model = StructuralCausalModel(direct_graph)
        self.direct_graph = direct_graph

        self.pre_processing = FunctionApplying() if pre_processing is None else pre_processing
        self.post_processing = FunctionApplying() if post_processing is None else post_processing

    def apply_processing(self, dataset: DataFrame, processing: FunctionApplying,
                         exclude: List[str] = None) -> DataFrame:
        """
        Applies a collection of transformation functions to a dataset, handling both augmentation
        and in-place replacement of columns based on the configuration of each function.

        Parameters:
        -----------
        dataset : DataFrame
            The dataset to be transformed.

        processing : FunctionApplying
            An object containing one or more transformation functions to apply row-wise.

        exclude: List[str], optional
            A list of column names to exclude from the transformation process. Defaults to None.
        Returns:
        --------
        DataFrame
            The transformed dataset, possibly with new columns added and/or original columns replaced
            according to the transformation logic.

        Notes:
        ------
        - Each transformation function is applied row-wise.
        - If a function has `replace=True`, the column(s) it depends on will be removed and the new
          column will be renamed to the original name.
        """
        if exclude is None:
            exclude = []

        processed = dataset.apply(axis=1, func=processing, exclude_fun=exclude)
        generated_data = pd.concat([dataset, processed], axis=1)

        for name, function in processing.functions.items():
            if function.replace and name not in exclude:
                generated_data.drop(columns=function.apply_on, inplace=True)
                generated_data.rename(
                    columns={function.feature_name: function.apply_on[0]},
                    inplace=True
                )
        return generated_data
    
    def auto_fit(self, dataset: DataFrame) -> Tuple:
        """
        Automatically assigns causal mechanisms based on the dataset and fits the model.

        Parameters:
        -----------
        dataset : DataFrame
            The dataset used to automatically assign and fit causal mechanisms.

        Returns:
        --------
        Tuple
            A tuple containing:
            - A summary of the automatic assignment process.
            - An evaluation summary of the model fit.
        """
        dataset = self.apply_processing(dataset=dataset,
                                        processing=self.pre_processing)

        summary_auto_assignment = assign_causal_mechanisms(
            causal_model=self.causal_model,
            based_on=dataset[list(self.direct_graph.nodes)],
            quality=AssignmentQuality.BETTER
        )

        # "fit" means "assign" a causal mechanism to the causal relationship
        evaluation_summary = gcm.fit(
            causal_model=self.causal_model,
            data=dataset,
            return_evaluation_summary=True
        )
        return summary_auto_assignment, evaluation_summary

    def custom_fit(self, dataset: DataFrame, causal_mechanisms: Dict[str, Mechanism]) -> Tuple:
        """
        Assigns specified mechanisms to variables in the causal model and fits it on the dataset.

        Parameters:
        -----------
        dataset : DataFrame
            The dataset used for fitting the model.

        causal_mechanisms : Dict[str, Mechanism]
            A dictionary mapping variable names to their custom causal mechanisms.

        Returns:
        --------
        Tuple
            A tuple containing:
            - None (since no auto-assignment is done).
            - An evaluation summary of the model fit.
        """
        dataset = self.apply_processing(dataset=dataset,
                                        processing=self.pre_processing)

        for feature, mechanism in causal_mechanisms.items():
            self.causal_model.set_causal_mechanism(feature, mechanism)

        evaluation_summary = gcm.fit(
            causal_model=self.causal_model,
            data=dataset,
            return_evaluation_summary=False
        )
        # summary_auto_assignment cannot be performed with custom causal assignment
        return None, evaluation_summary

    def sample(self,
               interventions: Dict[str, Callable] = None,
               observed_data: DataFrame = None,
               elements: int = None,
               apply_post_processing: bool = True,
               exclude_post_processing: List[str] = None,
               fill_missing: Literal["zero", "uniform"] = "zero",
               verbose: bool = False) -> DataFrame:
        """
        Generates synthetic data from the model, optionally under intervention, and applies post-processing.

        Parameters:
        -----------
        interventions : Dict, optional
            A dictionary of interventions, e.g., {"X": lambda x: value} to simulate do(X=value). Defaults to None.

        observed_data : DataFrame, optional
            A DataFrame containing observed data to apply the intervention. Defaults to None.

        elements : int
            The number of samples to generate.

        apply_post_processing : bool
            Whether to apply registered post-processing functions. Defaults to True.

        exclude_post_processing: List[str], optional
            A list of column names to exclude from the post-processing. Defaults to None.

        fill_missing: {"zero", "uniform"}, default="zero"
            Strategy used by `MechanismFromDict` when a parent configuration is missing:
            - "zero": Assign zero probability to all outcomes.
            - "uniform": Assign equal probability to all outcomes.

        verbose : bool, default=False
            Enables logging of information for missing or unexpected configurations
            during sampling in `MechanismFromDict`.
        Returns:
        --------
        DataFrame
            A DataFrame containing the synthetic samples (possibly post-processed).
        """
        if observed_data is None and elements is None:
            raise ValueError("Either observed_data or elements must be provided.")

        if observed_data is not None and len(observed_data) < 1:
            raise ValueError("The observed_data must contain at least one sample.")

        if observed_data is not None and elements is not None:
            elements = None
            warnings.warn("Since observed_data is provided, the number of elements will be ignored. ")

        if elements is not None and elements < 1:
            raise ValueError("The number of elements to generate must be at least one.")

        if elements is not None and observed_data is not None:
            observed_data = None
            warnings.warn("Since elements is provided, the observed_data will be ignored. ")

        if interventions is None and observed_data is not None:
            observed_data = None
            warnings.warn("Since no interventions are provided, the observed_data will be ignored ")
            if elements is None or elements < 1:
                raise ValueError("The number of elements to generate must be at least one.")

        if exclude_post_processing is None:
            exclude_post_processing = []

        for i, causal_mechanism in self.causal_model.graph.nodes(data=True):

            mechanism = causal_mechanism["causal_mechanism"]
            if isinstance(mechanism, ConditionalMechanismFromDict):
                mechanism.fill_missing = fill_missing
                mechanism.verbose = verbose

        if interventions is None:
            generated_data = gcm.draw_samples(self.causal_model, num_samples=elements)
        else:
            generated_data = gcm.interventional_samples(self.causal_model,
                                                        num_samples_to_draw=elements,
                                                        observed_data=observed_data,
                                                        interventions=interventions)
        if apply_post_processing:
            synthetic_data = self.apply_processing(dataset=generated_data,
                                                   processing=self.post_processing,
                                                   exclude=exclude_post_processing)
        else:
            synthetic_data = generated_data

        return synthetic_data

    def sample_and_save(self,
                        folder: Union[Path, str],
                        name: str,
                        add_file_info: bool = True,
                        **kwargs) -> DataFrame:
        """
        Generates and optionally intervenes in synthetic samples, applies post-processing, and saves to JSON.

        Parameters:
        -----------
        folder : Union[Path, str]
            folder to save the generated data as a JSON file.

        name : str
            name of the generated data file (without extension).

        interventions : Dict, optional
            A dictionary specifying interventions (do-operations). Defaults to None.

        observed_data : DataFrame, optional
            A DataFrame containing observed data to apply the intervention. Defaults to None.

        elements : int
            Number of samples to generate.

        apply_post_processing : bool
            Whether to apply post-processing. Defaults to True.

        fill_missing: {"zero", "uniform"}, default="zero"
            Strategy used by `MechanismFromDict` when a parent configuration is missing:
            - "zero": Assign zero probability to all outcomes.
            - "uniform": Assign equal probability to all outcomes.

        verbose : bool, default=False
            Enables logging of information for missing or unexpected configurations
            during sampling in `MechanismFromDict`.

        add_file_info: bool
            Whether to add additional information on the generated dataset. Defaults to True.
        Returns:
        --------
        DataFrame
            The generated (and saved) dataset.
        """
        sdt = self.sample(**kwargs)

        self.save(sdt=sdt, folder=folder, name=name, add_file_info=add_file_info)
        return sdt

    def save(self, sdt: DataFrame, folder: Union[Path, str], name: str,
             add_file_info: bool = True):
        """
        Saves the generated data to a JSON file.

        Parameters:
        -----------
        sdt : DataFrame
            The generated synthetic data to be saved.

        folder : Union[Path, str]
            Path to save the generated data as a JSON file.

        name : str
            Name of the generated data file (without extension).

        add_file_info: bool
            Whether to add additional information on the generated dataset. Defaults to True.
        """
        if isinstance(folder, str):
            folder = Path(folder)

        folder.mkdir(parents=True, exist_ok=True)

        # Save the DataFrame as JSON
        path = folder / name
        sdt.to_json(path.with_suffix(".json"), orient="records", indent=2)

        if add_file_info:
            # Save unique values and their counts
            unique_values = {}
            for col in sdt.columns:
                try:
                    unique_values[col] = sdt[col].value_counts().to_dict()
                except TypeError:
                    print(f"Warning: Column '{col}' contains non-hashable types and will be skipped for "
                          f"unique value counting.")

            with open(folder / f"{name.replace(".json", "")}_unique_values.json", "w") as f:
                json.dump(unique_values, f, indent=2)

            # Save the causal graph
            text_graph = ""
            for a, b in self.direct_graph.edges():
                text_graph += f"{a} -> {b}\n"

            with open(folder / f"{name.replace('.json', '')}_graph.txt", "w") as f:
                f.write(text_graph)

        return

    def evaluate_causal_model(self, dataset: DataFrame):
        """
        Evaluates the causal model using the provided dataset.

        Parameters:
        -----------
            dataset (DataFrame): The dataset to evaluate the model against.

        """
        return gcm.evaluate_causal_model(self.causal_model, dataset,
                                         evaluate_causal_mechanisms=False,
                                         evaluate_invertibility_assumptions=False)

    def falsify(self, dataset: DataFrame, permutations: int):
        """
        Tests whether the assumed causal structure is consistent with the data using permutation falsification.

        Parameters:
        -----------
        dataset : DataFrame
            Dataset used for the falsification test.

        permutations : int
            Number of permutations used to assess significance.

        Returns:
        --------
        dict
            Results of the falsification test, including test statistics and visual output.
        """
        return falsify_graph(self.direct_graph, n_permutations=permutations, data=dataset,
                             plot_histogram=True)
