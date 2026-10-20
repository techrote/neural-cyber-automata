"""One commissioned Jolt part-graph candidate; no optional trainer import here."""

from .core import (FEATURE_NAMES, JOLT_PROFILE_SHA256, PART_IDS, GraphFrame,
                   GraphIdentity, action_vector, gate_primary, make_frame)


def register(registry):
    """Declare replaceable candidates; creating one explicitly imports Torch."""
    def graph_factory(**kwargs):
        from .model import GraphPolicy
        return GraphPolicy("graph", **kwargs)

    def no_message_factory(**kwargs):
        from .model import GraphPolicy
        return GraphPolicy("no_message", **kwargs)

    for name, factory in (("jolt-three-part-graph", graph_factory),
                          ("jolt-three-part-no-message", no_message_factory)):
        registry.register("cellular", name, 1, factory,
                          ("jolt", "per_part", "root_readout", "bounded_messages", "cpu"))


__all__ = ["FEATURE_NAMES", "JOLT_PROFILE_SHA256", "PART_IDS", "GraphFrame", "GraphIdentity",
           "action_vector", "gate_primary", "make_frame", "register"]
