"""
solution_efficiency_scoring
=================

A small pipeline that estimates the effectiveness of individual "solutions"
from historical data on combinations of solutions and their outcomes.

Modules:
    data_processing         -- DataProcessor: load raw data, build the masked
                                combination matrix and label arrays.
    logit_model             -- LogitModelPipeline: train, evaluate, extrapolate,
                                and persist a logistic-regression model.
    result_reproduction     -- CLI entry point tying the two together
                                (`python -m solution_efficiency_scoring.train ...`).
"""

from .data_processing import DataProcessor
from .logit_model import LogitModelPipeline, iterative_logistic_training

__all__ = ["DataProcessor", "LogitModelPipeline", "iterative_logistic_training"]

__version__ = "0.1.0"
