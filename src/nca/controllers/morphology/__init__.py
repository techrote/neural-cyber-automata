"""Gated morphology preparation; importing the contracts needs no ML runtime."""

from .core import (
    ACTION_NAMES, EDGE_NAMES, NODE_NAMES, ROOT_NAMES, MAX_PARTS, MAX_JOINTS,
    MorphologyAdapter, MorphologyFrame, MorphologyGoal, MorphologyIdentity,
    SemanticAction, StateFence, information_reach, map_action,
)

__all__ = [
    "ACTION_NAMES", "EDGE_NAMES", "NODE_NAMES", "ROOT_NAMES", "MAX_PARTS", "MAX_JOINTS",
    "MorphologyAdapter", "MorphologyFrame", "MorphologyGoal", "MorphologyIdentity",
    "SemanticAction", "StateFence", "information_reach", "map_action",
]
