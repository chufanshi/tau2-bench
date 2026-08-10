# Retail visual-evidence policy

These clauses extend the standard retail policy. They apply only to delivered
orders and have the precedence shown below. An agent must inspect the attached
photo, cross-check the depicted product against the order record, and use only
the remedy licensed by the first matching clause.

1. **Evidence mismatch.** If the depicted product does not match the ordered
   line item, call `create_ticket` with reason `evidence_mismatch`. Do not
   refund, replace, or issue a voucher.
2. **Inadequate evidence.** If the photo is blurry, occluded, too distant, or
   otherwise fails to establish a required predicate, request a targeted
   additional photo. Do not call a remedy tool yet.
3. **No damage.** If an adequate photo shows no damage to the unit or package,
   no damage remedy is owed.
4. **Packaging-only damage.** If the unit is intact and only its packaging is
   damaged, call `issue_voucher` with `percentage=15`. Do not refund or replace.
5. **Fragile unit breakage.** For a glass or ceramic item, if an adequate photo
   shows unit-body breakage and a legible serial, call `refund_order`. No return
   shipment is required.
6. **Other unit damage.** If an adequate photo shows unit-body damage and a
   legible serial outside clause 5, call `create_replacement`. If the serial is
   not established, request a targeted serial-plate photo first.

The four remedy tools are mutually exclusive for one order item.
