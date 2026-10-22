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
