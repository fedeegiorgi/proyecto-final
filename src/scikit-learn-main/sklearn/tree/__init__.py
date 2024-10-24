"""Decision tree based models for classification and regression."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from ._classes import (
    BaseDecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from ._combined_tree import DecisionTreeRegressorCombiner # New class
# from ._extended_class import ContinuedDecisionTreeRegressor # New class
from ._export import export_graphviz, export_text, plot_tree

__all__ = [
    "BaseDecisionTree",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreeClassifier",
    "ExtraTreeRegressor",
    "DecisionTreeRegressorCombiner", # New class
    "export_graphviz",
    "plot_tree",
    "export_text",
]
