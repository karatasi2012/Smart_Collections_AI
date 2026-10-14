# MFI SmartCollect AI

A Streamlit prototype for AI-assisted collections prioritisation: accounts-receivable overview, overdue ageing, a baseline late-payment model, an explainable worklist, data-quality checks and CSV export.

> **Demo safety:** The bundled dataset is synthetic and is not MFI Group data. This is a prototype, not a production collections decision system. Scores must be reviewed by a collections/finance owner. No ERP connection or credentials are included.

## What's in the app

| Tab | Contents |
|---|---|
| Overview | Open and overdue KPIs, ageing chart, balance by business unit, top overdue customers |
| Collections worklist | Priority-ranked open invoices, filters, CSV export, per-invoice score drivers |
| Model | Hold-out metrics, comparison with a `historical_late_rate` baseline, score distribution, caveats |
| Data quality | Basic integrity checks and a data preview |
| About | Schema and pilot guidance |

Sidebar controls: CSV upload, "Shift demo dates to today" (keeps the synthetic demo looking current), and adjustable risk-band thresholds.

## Project layout

```
app.py                         Streamlit app
generate_sample_data.py        Reproducible synthetic data generator
data/demo_invoices.csv         Synthetic demo data (snapshot 2026-10-01)
sql/data_dictionary.md         Data contract
sql/normalized_views_template.sql  Starter SQL view (needs mapping to the real ERP schema)
tests/test_app.py              Smoke tests
.streamlit/config.toml         Theme and server settings
.github/workflows/ci.yml       Runs the tests on every push
requirements.txt               Runtime dependencies
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository (e.g. `MFI_SmartCollect_AI`) and upload the **contents** of this folder to the repository root, keeping the folder structure above (including the hidden `.streamlit` and `.github` folders).
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub and choose **Create app**.
3. Select the repository and branch, set **Main file path** to `app.py`, and deploy.

Never commit real customer data, credentials, connection strings or `.streamlit/secrets.toml`. Keep the repository private if company policy requires it.

## Run locally

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Run the tests with `python -m pip install -r requirements-dev.txt && python -m pytest`.

Regenerate the demo data with `python generate_sample_data.py`.

## Use your own validated CSV

Required columns: `invoice_id`, `customer_id`, `invoice_date`, `due_date`, `snapshot_date`, `invoice_amount`, `amount_paid`, `outstanding_amount`.

Optional: `fully_paid_date`, `payment_terms_days`, `dispute_flag`, `credit_note_amount`, `currency`, `business_unit`, `historical_late_rate`, `unpaid_30_days_after_due`. See [`sql/data_dictionary.md`](sql/data_dictionary.md).

The label `unpaid_30_days_after_due` must be reconstructed from payment-allocation history (`1` if a balance remained 30 days after the due date, `0` if settled). The app trains only on invoices whose 30-day window has elapsed at the snapshot date. Every feature must have been known at prediction time; the app excludes final `amount_paid`, `outstanding_amount` and `fully_paid_date` from the model for this reason.

## Limitations

- Read-only and CSV-first; no connection to Sage Evolution or SQL Server.
- Baseline logistic regression, not an autonomous agent and not a guarantee of payment.
- Synthetic labels demonstrate the workflow only; `historical_late_rate` is a noisy proxy for the hidden driver of the synthetic label.
- Risk bands and priority weights are demo values, not approved business rules.
- Amounts from multiple currencies are not converted; the app warns when it sees more than one.

## Before a live pilot

1. Agree definitions and a read-only data source with Finance and ERP owners.
2. Map real Sage Evolution fields to the schema; do not guess table or field names.
3. Reconcile invoice counts, balances, credits, write-offs and allocations to Sage reports.
4. Build point-in-time features and a time-based holdout; compare with a days-past-due baseline.
5. Agree acceptance thresholds, access controls, audit logging, monitoring, retention and human approval of collection actions.
6. Deploy only in an environment approved by MFI Group IT/security. Streamlit Community Cloud suits a synthetic-data demo, not automatically company financial data.
