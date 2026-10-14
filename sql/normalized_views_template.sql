/*
MFI SmartCollect — SQL Server normalization template
----------------------------------------------------
This is intentionally a TEMPLATE, not executable against an unverified Sage Evolution schema.
Map actual Sage Evolution 200 tables/fields with the ERP/database owner.

Create a read-only reporting view that returns ONE ROW PER INVOICE with the columns below.
Do not join raw invoice lines to raw receipts without aggregating first, or balances may multiply.
Validate all totals against Sage reports before using the output.
*/

-- Example normalized output contract (replace source objects/columns):
-- CREATE VIEW reporting.vw_SmartCollectInvoiceSnapshot AS
-- WITH InvoiceBase AS (
--     SELECT
--         <invoice_primary_key> AS invoice_id,
--         <customer_key> AS customer_id,
--         CAST(<invoice_date> AS date) AS invoice_date,
--         CAST(<due_date> AS date) AS due_date,
--         CAST(<original_invoice_total> AS decimal(19,4)) AS invoice_amount,
--         CAST(<snapshot_date_or_asof_date> AS date) AS snapshot_date,
--         <payment_terms_days> AS payment_terms_days,
--         <currency_code> AS currency,
--         <business_unit_or_country> AS business_unit
--     FROM <approved_invoice_source>
--     WHERE <exclude_voided_or_cancelled_document_condition>
-- ),
-- AllocatedPayments AS (
--     SELECT
--         <invoice_primary_key> AS invoice_id,
--         SUM(<allocated_payment_amount>) AS amount_paid
--     FROM <approved_receipt_allocation_source>
--     WHERE <exclude_reversed_or_unposted_allocations_condition>
--     GROUP BY <invoice_primary_key>
-- ),
-- Credits AS (
--     SELECT
--         <invoice_primary_key> AS invoice_id,
--         SUM(<allocated_credit_amount>) AS credit_note_amount
--     FROM <approved_credit_allocation_source>
--     GROUP BY <invoice_primary_key>
-- )
-- SELECT
--     i.invoice_id,
--     i.customer_id,
--     i.invoice_date,
--     i.due_date,
--     i.invoice_amount,
--     COALESCE(p.amount_paid, 0) AS amount_paid,
--     i.invoice_amount - COALESCE(p.amount_paid, 0) - COALESCE(c.credit_note_amount, 0) AS outstanding_amount,
--     i.snapshot_date,
--     <fully_paid_date_or_null> AS fully_paid_date,
--     i.payment_terms_days,
--     <dispute_flag_or_zero> AS dispute_flag,
--     COALESCE(c.credit_note_amount, 0) AS credit_note_amount,
--     i.currency,
--     i.business_unit
-- FROM InvoiceBase i
-- LEFT JOIN AllocatedPayments p ON p.invoice_id = i.invoice_id
-- LEFT JOIN Credits c ON c.invoice_id = i.invoice_id;

-- Required validation checks before use:
-- 1. Row count and invoice totals reconcile to Sage for the same as-of date.
-- 2. No duplicated invoice_id values in the output.
-- 3. Receipt allocations exclude unposted/reversed entries and are allocated to the right invoice.
-- 4. Credit notes, write-offs, cancellations, and multi-currency handling follow Finance policy.
-- 5. Historical labels reconstruct balance at due_date + 30 days from transaction/allocation history.
