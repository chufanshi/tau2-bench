"""Database state for retail visual-evidence claims."""

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from tau2.domains.retail.data_model import Order, RetailDB


VisualClaimResolution = Literal[
    "full_refund",
    "replacement",
    "shipping_voucher_15",
    "manual_review",
]


class RetailVisionOrder(Order):
    """Retail order with the two terminal statuses introduced by visual claims."""

    status: Literal[
        "processed",
        "pending",
        "pending (item modified)",
        "delivered",
        "cancelled",
        "exchange requested",
        "return requested",
        "refunded",
        "replacement requested",
    ]


class VisualClaim(BaseModel):
    """Deterministic environment record produced by a visual-policy action."""

    claim_id: str
    order_id: str
    item_id: str
    resolution: VisualClaimResolution
    requires_return: bool
    refund_amount: float = 0.0
    voucher_percent: int = 0
    replacement_order_id: Optional[str] = None
    review_reason: Optional[str] = None


class RetailVisionDB(RetailDB):
    """The original retail DB plus auditable visual-claim state."""

    orders: Dict[str, RetailVisionOrder]
    instance_serials: Dict[str, str] = Field(default_factory=dict)
    visual_claims: Dict[str, VisualClaim] = Field(default_factory=dict)

    @model_validator(mode="after")
    def populate_instance_serials(self) -> "RetailVisionDB":
        """Assign stable synthetic unit serials without mutating the base DB file."""

        expected = {
            f"{order_id}:{item.item_id}": f"TV-{item.item_id}"
            for order_id, order in self.orders.items()
            for item in order.items
        }
        if not self.instance_serials:
            self.instance_serials = expected
        elif self.instance_serials != expected:
            raise ValueError("retail-vision instance serial registry is inconsistent")
        return self
