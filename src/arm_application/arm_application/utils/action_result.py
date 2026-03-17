from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ActionResult:
    success: bool
    error_code: str = "OK"
    message: str = ""
    retryable: bool = False
    data: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def ok(message: str = "", data=None):
        return ActionResult(True, "OK", message, False, data or {})

    @staticmethod
    def fail(code: str, message: str, retryable: bool = False):
        return ActionResult(False, code, message, retryable, {})