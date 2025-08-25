# Tabular Data Generator with Structural Causal Models

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python Version](https://img.shields.io/badge/Python-%3E=3.12-blue.svg)](https://www.python.org/downloads/)

## Description

This framework provides a tool for generating synthetic tabular data based on Structural Causal Models (SCMs).
By utilizing SCMs, the framework allows for the explicit definition of causal relationships between variables,
ensuring that the generated data reflects these underlying dependencies. This approach is particularly useful for:

* **Realistic data generation:** Creating datasets that preserve the complex causal interactions present in real-world
  data.
* **Data augmentation:** Increasing the size and diversity of existing datasets while maintaining causal consistency.
* **Model testing and validation:** Generating controlled data to evaluate the behavior of machine learning algorithms
  in different scenarios.
* **Simulations:** Conducting "what-if" experiments and analyzing the consequences of interventions on variables.
* **Privacy-preserving data sharing:** Sharing synthetic data that retains important statistical characteristics without
  revealing sensitive information.

The framework offers a flexible interface for defining causal graphs, specifying the functions that describe the
relationships between variables, and generating datasets of arbitrary sizes.

# Citation
If you use this code or our results in your research, please cite our paper as follows:
```
@article{iommi2025,
  title={Causal Synthetic Data Generation in Recruitment},
  author={Iommi, Andrea and Mastropietro, Antonio and Guidotti, Riccardo and Monreale, Anna and Ruggieri, Salvatore},
  year={2025}
}
```
