"""Cost guard: every paid API call in PhysicsLens goes through CostGuardClient."""

from .client import CallResult, CostGuardClient
from .errors import (
    BudgetExceeded,
    CacheOnlyMiss,
    CircuitOpen,
    CostGuardError,
    InvalidRequest,
    KillSwitchActive,
    ModelNotAllowed,
    OutputTruncated,
    PriceMissing,
    ProviderCallFailed,
)
from .pricing import Usage

__all__ = [
    "BudgetExceeded",
    "CacheOnlyMiss",
    "CallResult",
    "CircuitOpen",
    "CostGuardClient",
    "CostGuardError",
    "InvalidRequest",
    "KillSwitchActive",
    "ModelNotAllowed",
    "OutputTruncated",
    "PriceMissing",
    "ProviderCallFailed",
    "Usage",
]
