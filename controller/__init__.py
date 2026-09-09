from __future__ import annotations

"""Phase 1 MLP evolution controller: state encoding, dataset, controllers."""

from controller.dataset import (
    build_multihead_samples,
    build_supervised_samples,
    load_trajectories,
    merge_state_reward,
    train_val_split,
)
from controller.mlp_controller import ConstantController, MLPController
from controller.multihead_controller import MultiHeadController
from controller.problem_features import PROBLEM_FEATURE_NAMES, problem_feature_vector
from controller.state_encoder import STATE_FEATURES, ProblemAwareEncoder, StateEncoder

__all__ = [
    "STATE_FEATURES",
    "StateEncoder",
    "ProblemAwareEncoder",
    "PROBLEM_FEATURE_NAMES",
    "problem_feature_vector",
    "load_trajectories",
    "merge_state_reward",
    "build_supervised_samples",
    "build_multihead_samples",
    "train_val_split",
    "MLPController",
    "ConstantController",
    "MultiHeadController",
]
