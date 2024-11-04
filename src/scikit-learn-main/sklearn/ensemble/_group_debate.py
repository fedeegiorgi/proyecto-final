from sklearn.ensemble import RandomForestRegressor
import threading
import numpy as np
from itertools import islice
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import issparse

from ..base import (
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
    is_classifier,
)
from ..exceptions import DataConversionWarning
from ..metrics import accuracy_score, r2_score
from ..preprocessing import OneHotEncoder
from ..tree import (
    BaseDecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from ..tree._tree import DOUBLE, DTYPE
from ..utils import check_random_state, compute_sample_weight
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils._tags import _safe_tags
from ..utils.multiclass import check_classification_targets, type_of_target
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
    _check_feature_names_in,
    _check_sample_weight,
    _num_samples,
    check_is_fitted,
)
from ._base import BaseEnsemble, _partition_estimators

class RandomForestGroupDebate(RandomForestRegressor):
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
        group_size=10,  # New parameter
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,
        )

        # Initialize the new parameter specific to this class
        self.group_size = group_size

        if self.n_estimators % self.group_size == 0 and self.n_estimators > self.group_size:
            self._n_groups = int(self.n_estimators / self.group_size)
        else:
            raise ValueError("La # árboles mod group_size diferente que 0 o la # árboles es menor o igual al group_size")

    def random_group_split(self, predictions):
        n_samples = predictions.shape[1]
        
        if self.n_outputs_ > 1:
            # Multi-output prediction: ver después si implementamos esto (sería transformar matrix 2D a 3D)
            pass
        else:
            # Reshape the array into a 3D tensor: (num_groups, group_size, n_samples)
            tree_groups_tensor = predictions.reshape(self._n_groups, self.group_size, n_samples)

        return tree_groups_tensor

    def group_split(self, iterable):
        # Create a list of lists by slicing the iterable into groups of size `self._n_groups`
        grouped = []
        for i in range(self._n_groups):
            lower_bound = i * self.group_size
            upper_bound = (i + 1) * self.group_size
            group = iterable[lower_bound:upper_bound]
            grouped.append(group)
        return grouped