"""Every way costguard can refuse or abort a paid call."""


class CostGuardError(RuntimeError):
    """Base class: a paid call was refused or failed. Message is safe to show in the UI."""


class KillSwitchActive(CostGuardError):
    pass


class ModelNotAllowed(CostGuardError):
    pass


class PriceMissing(CostGuardError):
    pass


class BudgetExceeded(CostGuardError):
    pass


class CircuitOpen(CostGuardError):
    pass


class InvalidRequest(CostGuardError):
    """Our own request violates a cost rule (e.g. a forbidden reasoning effort)."""


class ProviderCallFailed(CostGuardError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class OutputTruncated(CostGuardError):
    """The model hit its output-token limit. The call was paid for; raise the limit, don't just retry."""
