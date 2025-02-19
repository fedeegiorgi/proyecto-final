"""Ensemble-based methods for classification, regression and anomaly detection."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from ._bagging import BaggingClassifier, BaggingRegressor
from ._base import BaseEnsemble
from ._forest import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    RandomTreesEmbedding,
)
from ._group_debate import(
    RandomForestGroupDebate,
)
from ._extremos_forest import (
    IQRRandomForestRegressor,
    PercentileTrimmingRandomForestRegressor,
)
from ._oob_forest import (
    OOBRandomForestRegressor,
    OOBPlusIQRRandomForestRegressor,
)
from ._first_split_combiner import (
    FirstSplitCombinerRandomForestRegressor,
)
from ._shared_knowledge import (
    SharedKnowledgeRandomForestRegressor,
)
from ._gb import GradientBoostingClassifier, GradientBoostingRegressor
from ._hist_gradient_boosting.gradient_boosting import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from ._iforest import IsolationForest
from ._stacking import StackingClassifier, StackingRegressor
from ._voting import VotingClassifier, VotingRegressor
from ._weight_boosting import AdaBoostClassifier, AdaBoostRegressor

__all__ = [
    "BaseEnsemble",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "IQRRandomForestRegressor",
    "PercentileTrimmingRandomForestRegressor",
    "OOBRandomForestRegressor",
    "OOBPlusIQRRandomForestRegressor",
    "FirstSplitCombinerRandomForestRegressor",
    "SharedKnowledgeRandomForestRegressor",
    "RandomTreesEmbedding",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "BaggingClassifier",
    "BaggingRegressor",
    "IsolationForest",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "VotingClassifier",
    "VotingRegressor",
    "StackingClassifier",
    "StackingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
]
