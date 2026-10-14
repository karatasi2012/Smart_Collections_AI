# Normalized invoice data contract

The dashboard expects one row per invoice. For partial payments, `amount_paid` is the total amount allocated to the invoice as of `snapshot_date`; `outstanding_amount` should be calculated as invoice amount minus allocated payments/credits according to Finance-approved rules.

| Field | Type | Required | Meaning |
|---|---|---:|---|
| invoice_id | string | Yes | Unique invoice identifier |
| customer_id | string | Yes | Stable customer identifier |
| invoice_date | date | Yes | Invoice posting/document date |
| due_date | date | Yes | Contractual due date |
| invoice_amount | decimal | Yes | Original invoice amount, positive for invoices |
| amount_paid | decimal | Yes | Payments allocated to invoice as of snapshot |
| outstanding_amount | decimal | Yes | Remaining balance at snapshot date |
| snapshot_date | date | Yes | Date at which amounts/status were observed |
| fully_paid_date | date or blank | No | Date invoice was fully settled, if available |
| payment_terms_days | integer | No | Due-date interval or approved payment term |
| dispute_flag | 0/1 | No | Whether a dispute was recorded at snapshot |
| credit_note_amount | decimal | No | Credit notes allocated to the invoice |
| currency | string | No | Currency code |
| business_unit | string | No | Country/business unit for filtering |

## Target creation

For invoices whose due date + 30 days is before the snapshot date:
- `unpaid_30_days_after_due = 1` if balance at due date + 30 days was greater than the agreed tolerance.
- `unpaid_30_days_after_due = 0` if the invoice was fully settled by that point.

For accurate labels, use payment allocation history to reconstruct balance at the 30-day point. Do not derive historical outcomes from today's outstanding balance alone if invoices may have been paid later, credited, written off, or reversed.

## Feature timing / leakage control

Every model feature must have been available at the prediction date. Do not use eventual payment date, eventual final balance, write-off outcome, or any field updated after the prediction date as a feature. The sample prototype uses invoice-level historical fields as a demonstration; production training should build point-in-time snapshots.
