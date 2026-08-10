"""Environment-changing actions licensed by the retail visual policy."""

from tau2.domains.retail.data_model import OrderItem, OrderPayment
from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail_vision.data_model import (
    RetailVisionDB,
    RetailVisionOrder,
    VisualClaim,
    VisualClaimResolution,
)
from tau2.environment.toolkit import ToolType, is_tool


class RetailVisionTools(RetailTools):
    """Standard retail tools plus four mutually exclusive visual remedies."""

    db: RetailVisionDB

    def __init__(self, db: RetailVisionDB) -> None:
        super().__init__(db)

    def _claim_item(
        self, order_id: str, item_id: str
    ) -> tuple[RetailVisionOrder, OrderItem]:
        order = self._get_order(order_id)
        if order.status != "delivered":
            raise ValueError("Visual claims require a delivered order")
        matches = [item for item in order.items if item.item_id == item_id]
        if len(matches) != 1:
            raise ValueError("Claim item must occur exactly once in the order")
        claim_id = f"visual_claim:{order_id}:{item_id}"
        if claim_id in self.db.visual_claims:
            raise ValueError("A visual claim already exists for this order item")
        return order, matches[0]

    def _record_claim(
        self,
        order: RetailVisionOrder,
        item: OrderItem,
        resolution: VisualClaimResolution,
        *,
        requires_return: bool,
        refund_amount: float = 0.0,
        voucher_percent: int = 0,
        replacement_order_id: str | None = None,
        review_reason: str | None = None,
    ) -> VisualClaim:
        claim_id = f"visual_claim:{order.order_id}:{item.item_id}"
        claim = VisualClaim(
            claim_id=claim_id,
            order_id=order.order_id,
            item_id=item.item_id,
            resolution=resolution,
            requires_return=requires_return,
            refund_amount=round(refund_amount, 2),
            voucher_percent=voucher_percent,
            replacement_order_id=replacement_order_id,
            review_reason=review_reason,
        )
        self.db.visual_claims[claim_id] = claim
        return claim

    @is_tool(ToolType.WRITE)
    def refund_order(self, order_id: str, item_id: str) -> VisualClaim:
        """Refund a delivered single-item order without requiring a return.

        Args:
            order_id: Delivered order containing the visually verified item.
            item_id: Unique line-item variant depicted by the evidence.

        Returns:
            The recorded full-refund visual claim.
        """
        order, item = self._claim_item(order_id, item_id)
        if len(order.items) != 1:
            raise ValueError("Full no-return refund currently requires a single-item order")
        payment_method_id = order.payment_history[0].payment_method_id
        order.payment_history.append(
            OrderPayment(
                transaction_type="refund",
                amount=item.price,
                payment_method_id=payment_method_id,
            )
        )
        order.status = "refunded"
        return self._record_claim(
            order,
            item,
            "full_refund",
            requires_return=False,
            refund_amount=item.price,
        )

    @is_tool(ToolType.WRITE)
    def create_replacement(self, order_id: str, item_id: str) -> VisualClaim:
        """Create a no-charge replacement request for a delivered item.

        Args:
            order_id: Delivered order containing the visually verified item.
            item_id: Unique line-item variant to replace.

        Returns:
            The recorded replacement visual claim.
        """
        order, item = self._claim_item(order_id, item_id)
        replacement_id = f"#R{order_id.removeprefix('#W')}"
        order.status = "replacement requested"
        return self._record_claim(
            order,
            item,
            "replacement",
            requires_return=False,
            replacement_order_id=replacement_id,
        )

    @is_tool(ToolType.WRITE)
    def issue_voucher(
        self, order_id: str, item_id: str, percentage: int = 15
    ) -> VisualClaim:
        """Issue the fixed shipping voucher for packaging-only damage.

        Args:
            order_id: Delivered order containing the affected item.
            item_id: Unique line-item variant whose packaging was damaged.
            percentage: Voucher percentage; visual policy fixes this at 15.

        Returns:
            The recorded voucher visual claim.
        """
        if percentage != 15:
            raise ValueError("Visual policy permits only a 15 percent voucher")
        order, item = self._claim_item(order_id, item_id)
        return self._record_claim(
            order,
            item,
            "shipping_voucher_15",
            requires_return=False,
            voucher_percent=percentage,
        )

    @is_tool(ToolType.WRITE)
    def create_ticket(
        self, order_id: str, item_id: str, reason: str = "evidence_mismatch"
    ) -> VisualClaim:
        """Open manual review when evidence does not match the order item.

        Args:
            order_id: Delivered order associated with the claim.
            item_id: Ordered line item that the evidence failed to match.
            reason: Review reason; must be evidence_mismatch.

        Returns:
            The recorded manual-review visual claim.
        """
        if reason != "evidence_mismatch":
            raise ValueError("Visual-policy tickets require evidence_mismatch")
        order, item = self._claim_item(order_id, item_id)
        return self._record_claim(
            order,
            item,
            "manual_review",
            requires_return=False,
            review_reason=reason,
        )

