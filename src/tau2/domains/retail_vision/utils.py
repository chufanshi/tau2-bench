"""Paths for the retail-vision domain."""

from tau2.domains.retail.utils import RETAIL_DB_PATH
from tau2.utils.utils import DATA_DIR

RETAIL_VISION_DATA_DIR = DATA_DIR / "tau2" / "domains" / "retail_vision"
RETAIL_VISION_POLICY_PATH = RETAIL_VISION_DATA_DIR / "visual_policy.md"
RETAIL_VISION_TASK_SET_PATH = RETAIL_VISION_DATA_DIR / "tasks.json"

__all__ = [
    "RETAIL_DB_PATH",
    "RETAIL_VISION_POLICY_PATH",
    "RETAIL_VISION_TASK_SET_PATH",
]

