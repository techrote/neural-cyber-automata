"""Shared syntax checks for a separately commissioned v2 allocation identity.

These checks perform no GitHub lookup and grant no authorization. The cited
owner commission and historical uniqueness must be reviewed independently
before any declaration or invocation is admitted.

© 2026 techrote. All rights reserved.
"""

from __future__ import annotations

from collections.abc import Mapping
import re

from nca.contracts import ContractError


def validate_v2_allocation_identity(declaration):
    """Return the bounded identity fields after validating their exact syntax."""
    if not isinstance(declaration, Mapping):
        raise ContractError("jolt_allocation_identity", "a versioned allocation identity object is required")
    allocation_id = declaration.get("allocation_id")
    authorization_ref = declaration.get("authorization_ref")
    if (not isinstance(allocation_id, str) or not 1 <= len(allocation_id) <= 80
            or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", allocation_id) is None):
        raise ContractError("jolt_allocation_identity", "a bounded lowercase allocation slug is required")
    if (not isinstance(authorization_ref, str)
            or re.fullmatch(r"https://github\.com/techrote/neural-cyber-automata/issues/62#issuecomment-[1-9][0-9]{0,19}",
                            authorization_ref) is None):
        raise ContractError("jolt_allocation_identity", "an exact public issue 62 commissioning comment reference is required")
    return {"allocation_id": allocation_id, "authorization_ref": authorization_ref}


def require_prospective_review(record):
    """An offline software review cannot silently admit a physical experiment."""
    if (not isinstance(record, Mapping) or record.get("offline_scope_only") is not False
            or record.get("prospective_experiment_admitted") is not True):
        raise ContractError("jolt_execution_review_scope",
                            "v2 execution needs explicit prospective experiment review")


def validate_v3_allocation_identity(declaration, *, consumed_allocation_ids=()):
    """Validate the new experiment identity against an independently bound registry.

    The default empty registry does not establish historical freshness. Final
    prospective admission must supply the reviewed consumed-allocation inventory;
    runtime also refuses every already-created output/lifecycle identity.
    """
    if (not isinstance(declaration, Mapping)
            or declaration.get("schema") != "nca-learn-jolt-003-execution/v3"):
        raise ContractError("jolt_v3_allocation_identity", "the separately versioned LEARN-JOLT-003 declaration is required")
    identity = validate_v2_allocation_identity(declaration)
    if not identity["allocation_id"].startswith("learn-jolt-003-"):
        raise ContractError("jolt_v3_allocation_identity", "the allocation must name the new LEARN-JOLT-003 experiment")
    if identity["allocation_id"] in consumed_allocation_ids:
        raise ContractError("jolt_v3_allocation_consumed", "a consumed allocation cannot be admitted or resumed")
    return identity
